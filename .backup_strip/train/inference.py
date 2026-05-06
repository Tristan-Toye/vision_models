"""Inference module for zombie detection.

Provides a unified ZombieDetector class that loads any trained model
and returns bounding boxes in the format expected by evaluation.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from train.models import get_model, HEATMAP_MODELS, SELF_TRAINING_MODELS
from train.models.heatmap_cnn import heatmap_to_boxes
from train.preprocessing import NUM_MANUAL_FEATURES, build_preprocessing_pipeline


PACKAGE_DIR = Path(__file__).resolve().parent


def resolve_inference_device(device: Optional[str]) -> str:
    """Resolve ``device`` for inference: pick GPU when available, else CPU (or MPS on macOS).

    * ``None``, ``\"\"``, ``\"auto\"``, ``\"auto_detect\"`` — use CUDA if
      :func:`torch.cuda.is_available`, else MPS if available, else ``\"cpu\"``.
    * ``\"cuda\"`` / ``\"gpu\"`` — CUDA when available, otherwise fall back to CPU.
    * ``\"mps\"`` — Apple MPS when available, else same as ``\"auto\"``.
    * ``\"cpu\"`` — always CPU.
    """
    if device is None:
        d = "auto"
    else:
        d = str(device).strip().lower()
    if d in ("", "auto", "auto_detect"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if d in ("cuda", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return resolve_inference_device("auto")
    return "cpu"


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = str(PACKAGE_DIR / "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_weights_path(model_type: str, model_path: str) -> str:
    """Resolve a checkpoint *file* path from a file or experiment directory."""
    p = Path(model_path).expanduser().resolve()
    if p.is_file():
        return str(p)
    if not p.is_dir():
        return str(p)

    if model_type in {"yolov8n", "yolov11n", "rt_detr"}:
        cands = sorted(p.glob("**/weights/best.pt"))
        if cands:
            return str(cands[0])
        raise FileNotFoundError(f"No weights/best.pt under {p}")

    if model_type == "hog_svm":
        hp = p / "hog_svm.joblib"
        if hp.exists():
            return str(hp)
        raise FileNotFoundError(f"No hog_svm.joblib under {p}")

    if model_type == "template_match":
        # Directory mode: template saved at train time
        return str(p)

    for name in ("best_model.pt", "last_model.pt"):
        fp = p / name
        if fp.exists():
            return str(fp)
    raise FileNotFoundError(f"No best_model.pt / last_model.pt under {p}")


def load_train_settings(model_path: str) -> dict:
    """Load ``train_settings.json`` beside the checkpoint (walk up for YOLO layout)."""
    defaults: dict[str, Any] = {
        "resize": None,
        "preprocessing_variant": "rgb",
        "manual_features": False,
    }
    cur = Path(model_path).expanduser().resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(6):
        settings_path = cur / "train_settings.json"
        if settings_path.exists():
            with open(settings_path, "r") as f:
                defaults.update(json.load(f))
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    return defaults


def infer_input_channels(settings: dict) -> int:
    pv = settings.get("preprocessing_variant") or "rgb"
    mf = bool(settings.get("manual_features", False))
    ch = 4 if pv == "rgb_edge" else 3
    if mf:
        ch += NUM_MANUAL_FEATURES
    return ch


class ZombieDetector:
    """Unified inference interface for all zombie detection models.

    Args:
        model_path: path to weights file **or** experiment directory containing
            ``best_model.pt`` / Ultralytics ``weights/best.pt`` / ``hog_svm.joblib``.
        model_type: model name from the registry (e.g., ``yolov8n``, ``heatmap_cnn``).
        config_path: path to config.yaml.
        preprocessing_variant: ``rgb``, ``grayscale``, ``rgb_edge``, or ``auto``
            (read from ``train_settings.json`` when trained with the unified trainer).
        device: ``cuda``, ``cpu``, ``mps``, or ``auto`` (pick best at runtime).
        conf_threshold: confidence threshold for detections / heatmap peaks.
        resize: ``[h, w]``, ``None``, or ``auto`` (from ``train_settings.json``).
    """

    def __init__(
        self,
        model_path: str,
        model_type: str,
        config_path: Optional[str] = None,
        preprocessing_variant: str = "rgb",
        device: str = "auto",
        conf_threshold: float = 0.3,
        resize: "list[int] | None | str" = "auto",
    ):
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = resolve_inference_device(device)
        self.cfg = load_config(config_path)

        self.bbox_w = self.cfg["zombie"]["bbox_width"]
        self.bbox_h = self.cfg["zombie"]["bbox_height"]
        self.screen_w = self.cfg["screen"]["width"]
        self.screen_h = self.cfg["screen"]["height"]

        self._weights_path = resolve_weights_path(model_type, model_path)
        settings = load_train_settings(self._weights_path)

        if resize == "auto":
            resize = settings.get("resize")
            if resize is None:
                resize = self.cfg["preprocessing"].get("resize")

        if preprocessing_variant == "auto":
            preprocessing_variant = settings.get("preprocessing_variant") or "rgb"

        self.manual_features = bool(settings.get("manual_features", False))
        self.resize = resize
        self.preprocessing_variant = preprocessing_variant

        self.transform = build_preprocessing_pipeline(
            self.cfg,
            variant=preprocessing_variant,
            augment=False,
            resize=resize,
            manual_features=self.manual_features,
        )

        self._settings = settings
        self.model = self._load_model()

    def _load_model(self):
        if self.model_type in {"yolov8n", "yolov11n", "rt_detr"}:
            return get_model(self.model_type, checkpoint=self._weights_path)

        if self.model_type == "hog_svm":
            return get_model(self.model_type, checkpoint=self._weights_path)

        if self.model_type == "template_match":
            wp = Path(self._weights_path)
            if wp.suffix.lower() == ".npy" and wp.is_file():
                return get_model(self.model_type, template_path=str(wp))
            root = wp if wp.is_dir() else wp.parent
            tpl = root / "template_gray.npy"
            if tpl.exists():
                return get_model(self.model_type, template_path=str(tpl))
            return get_model(self.model_type)

        in_ch = infer_input_channels(
            {
                "preprocessing_variant": self.preprocessing_variant,
                "manual_features": self.manual_features,
            },
        )
        model_kwargs: dict[str, Any] = {"in_channels": in_ch}

        if self.model_type in {"resnet18_head", "resnet50_head"}:
            model_kwargs["pretrained"] = False
        if self.model_type == "faster_rcnn":
            model_kwargs["pretrained"] = False
            model_kwargs["freeze_backbone"] = False

        model = get_model(self.model_type, **model_kwargs)

        if isinstance(model, torch.nn.Module) and Path(self._weights_path).exists():
            map_loc = "cpu" if self.device == "mps" else self.device
            try:
                state = torch.load(self._weights_path, map_location=map_loc, weights_only=True)
            except TypeError:
                state = torch.load(self._weights_path, map_location=map_loc)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()

        return model

    def detect(self, observation: np.ndarray) -> np.ndarray:
        """Detect zombies in an observation image.

        Args:
            observation: (H, W, 3) uint8 image

        Returns:
            (N, 4) array of [x, y, w, h] bounding boxes in screen coords,
            ordered from most confident to least.
        """
        if self.model_type in {"yolov8n", "yolov11n", "rt_detr"}:
            return self._detect_ultralytics(observation)
        if self.model_type in HEATMAP_MODELS:
            return self._detect_heatmap(observation)
        if self.model_type == "faster_rcnn":
            return self._detect_fasterrcnn(observation)
        return self.model.predict(observation, conf=self.conf_threshold)

    def _detect_ultralytics(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image, conf=self.conf_threshold, device=self.device)

    def _detect_heatmap(self, image: np.ndarray) -> np.ndarray:
        boxes_dummy = np.zeros((0, 4), dtype=np.float32)
        image_proc, _ = self.transform(image, boxes_dummy)

        img_tensor = torch.from_numpy(image_proc).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            heatmap = self.model(img_tensor)[0, 0].cpu().numpy()

        return heatmap_to_boxes(
            heatmap,
            self.bbox_w,
            self.bbox_h,
            self.screen_w,
            self.screen_h,
            threshold=self.conf_threshold,
        )

    def _detect_fasterrcnn(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image, conf=self.conf_threshold, device=self.device)
