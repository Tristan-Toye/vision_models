"""Inference module for zombie detection.

Provides a unified ZombieDetector class that loads any trained model
and returns bounding boxes in the format expected by evaluation.py.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from zombie_detection.models import get_model, HEATMAP_MODELS, SELF_TRAINING_MODELS
from zombie_detection.models.heatmap_cnn import heatmap_to_boxes
from zombie_detection.preprocessing import build_preprocessing_pipeline


PACKAGE_DIR = Path(__file__).resolve().parent


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = str(PACKAGE_DIR / "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ZombieDetector:
    """Unified inference interface for all zombie detection models.

    Args:
        model_path: path to the trained model checkpoint
        model_type: model name from the registry (e.g., 'yolov8n', 'heatmap_cnn')
        config_path: path to config.yaml
        preprocessing_variant: 'rgb', 'grayscale', or 'rgb_edge'
        device: 'cuda' or 'cpu'
        conf_threshold: confidence threshold for detections
        resize: [h, w] to resize input images, or None for native resolution.
                If not supplied, the detector looks for a train_settings.json
                beside the checkpoint to auto-detect the training resize.
    """

    def __init__(
        self,
        model_path: str,
        model_type: str,
        config_path: Optional[str] = None,
        preprocessing_variant: str = "rgb",
        device: str = "cpu",
        conf_threshold: float = 0.3,
        resize: "list[int] | None" = "auto",
    ):
        self.model_type = model_type
        self.conf_threshold = conf_threshold
        self.device = device
        self.cfg = load_config(config_path)

        self.bbox_w = self.cfg["zombie"]["bbox_width"]
        self.bbox_h = self.cfg["zombie"]["bbox_height"]
        self.screen_w = self.cfg["screen"]["width"]
        self.screen_h = self.cfg["screen"]["height"]

        # Resolve resize to match what the model was trained with
        if resize == "auto":
            resize = self._load_train_resize(model_path)

        self.resize = resize
        self.transform = build_preprocessing_pipeline(
            self.cfg, variant=preprocessing_variant, augment=False, resize=resize,
        )

        self.model = self._load_model(model_path)

    def _load_train_resize(self, model_path: str) -> "list[int] | None":
        """Try to read the resize setting from train_settings.json saved
        alongside the checkpoint during training."""
        settings_path = Path(model_path).parent / "train_settings.json"
        if settings_path.exists():
            with open(settings_path, "r") as f:
                settings = json.load(f)
            return settings.get("resize")
        # Fallback: use config default
        return self.cfg["preprocessing"].get("resize")

    def _load_model(self, model_path: str):
        if self.model_type in SELF_TRAINING_MODELS:
            return get_model(self.model_type, checkpoint=model_path)

        in_channels = 3
        model_kwargs = {"in_channels": in_channels}

        if self.model_type in {"resnet18_head", "resnet50_head"}:
            model_kwargs["pretrained"] = False

        model = get_model(self.model_type, **model_kwargs)

        if isinstance(model, torch.nn.Module) and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device, weights_only=True)
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
        if self.model_type in SELF_TRAINING_MODELS:
            return self._detect_ultralytics(observation)
        elif self.model_type in HEATMAP_MODELS:
            return self._detect_heatmap(observation)
        elif self.model_type == "faster_rcnn":
            return self._detect_fasterrcnn(observation)
        else:
            # Classical models
            return self.model.predict(observation, conf=self.conf_threshold)

    def _detect_ultralytics(self, image: np.ndarray) -> np.ndarray:
        return self.model.predict(image, conf=self.conf_threshold)

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
