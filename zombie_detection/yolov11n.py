from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zombie_detection.interface import ZombieDetectorPipeline


def _repo_root() -> Path:
    # .../vision_models/zombie_detection/yolov11n.py -> repo root
    return Path(__file__).resolve().parents[1]


def _resolve_device(device: str) -> str:
    d = (device or "auto").strip().lower()
    if d not in {"auto", "cpu", "cuda", "gpu", "mps"}:
        d = "auto"
    if d == "cpu":
        return "cpu"
    if d in {"cuda", "gpu"}:
        return "cuda"
    if d == "mps":
        return "mps"

    # auto
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class YOLOv11nPipeline(ZombieDetectorPipeline):
    """Runtime-only YOLOv11n detector (fixed model selection).

    Loads weights from:
      zombie_detection/yolov11n/weights/best.pt
    """

    device: str = "auto"
    conf_threshold: float = 0.35

    def __post_init__(self) -> None:
        self._device = _resolve_device(self.device)
        weights = (
            _repo_root()
            / "zombie_detection"
            / "yolov11n"
            / "weights"
            / "best.pt"
        )
        if not weights.is_file():
            raise FileNotFoundError(f"Missing YOLOv11n weights: {weights}")

        from ultralytics import YOLO

        self._weights = weights
        self._model = YOLO(str(weights))

    @property
    def pipeline_id(self) -> str:
        return "yolov11n"

    def detect(self, observation: np.ndarray) -> np.ndarray:
        results = self._model.predict(
            observation,
            conf=float(self.conf_threshold),
            device=self._device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        boxes = np.zeros((len(xyxy), 4), dtype=np.float32)
        boxes[:, 0] = xyxy[:, 0]
        boxes[:, 1] = xyxy[:, 1]
        boxes[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        boxes[:, 3] = xyxy[:, 3] - xyxy[:, 1]

        order = np.argsort(-confs)
        return boxes[order]

    def describe(self) -> str:
        return f"yolov11n device={self._device} weights={self._weights}"

