"""RT-DETR wrapper using Ultralytics.

Transformer-based end-to-end detector: no NMS needed, uses
learned object queries to predict detections directly.
"""

from pathlib import Path
from typing import Optional

import numpy as np


class RTDETRDetector:
    """Wraps Ultralytics RT-DETR for training and inference."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        **kwargs,
    ):
        self.checkpoint = checkpoint
        self.model = None

    def _load_model(self):
        from ultralytics import RTDETR
        if self.checkpoint and Path(self.checkpoint).exists():
            self.model = RTDETR(self.checkpoint)
        else:
            self.model = RTDETR("rtdetr-l.pt")

    def train(
        self,
        dataset_yaml: str,
        epochs: int = 50,
        batch_size: int = 16,
        imgsz: int = 640,
        project: str = "zombie_detection/runs",
        name: str = "rtdetr_train",
        **train_kwargs,
    ) -> dict:
        if self.model is None:
            self._load_model()

        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            verbose=True,
            **train_kwargs,
        )
        return results

    def predict(self, image: np.ndarray, conf: float = 0.25) -> np.ndarray:
        """Run inference on a single image (H, W, 3) uint8.

        Returns (N, 4) array of [x, y, w, h] in pixel coords.
        """
        if self.model is None:
            self._load_model()

        results = self.model.predict(image, conf=conf, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
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

    def evaluate(self, dataset_yaml: str, split: str = "test") -> dict:
        if self.model is None:
            self._load_model()
        return self.model.val(data=dataset_yaml, split=split)
