"""Heatmap CNN — ``submission_models/heatmap_cnn/best_model.pt`` (+ ``train_settings.json``).

Best mixed test precision: **0.9848** (mse, rgb, manual features, 360×640).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from zombie_detection.inference import ZombieDetector
from zombie_detection.realtime.deployment._paths import submission_models_dir
from zombie_detection.realtime.interface import ZombieDetectorPipeline


class HeatmapCNNDeploymentPipeline(ZombieDetectorPipeline):
    mixed_test_precision: float = 0.9847567941317942
    _local_dir: Path = submission_models_dir() / "heatmap_cnn"

    @property
    def pipeline_id(self) -> str:
        return "heatmap_cnn"

    def __init__(
        self,
        *,
        device: str = "auto",
        conf_threshold: float = 0.35,
        config_path: Optional[str] = None,
    ):
        if not self._local_dir.is_dir():
            raise FileNotFoundError(
                f"Missing submission model directory: {self._local_dir}. "
                "Run zombie_detection/realtime/deployment/sync_submission_models.py",
            )
        self._detector = ZombieDetector(
            model_path=str(self._local_dir),
            model_type="heatmap_cnn",
            config_path=config_path,
            device=device,
            conf_threshold=conf_threshold,
            resize="auto",
            preprocessing_variant="auto",
        )

    def detect(self, observation: np.ndarray) -> np.ndarray:
        return self._detector.detect(observation)

    def describe(self) -> str:
        return (
            f"heatmap_cnn mixed_test_precision={self.mixed_test_precision:.4f} "
            f"device={self._detector.device} dir={self._local_dir}"
        )
