"""Faster R-CNN — ``submission_models/faster_rcnn/``.

Not shipped in the default submission registry (large + slower), but useful for
latency profiling and debugging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from zombie_detection.inference import ZombieDetector
from zombie_detection.realtime.deployment._paths import submission_models_dir
from zombie_detection.realtime.interface import ZombieDetectorPipeline


class FasterRCNNDeploymentPipeline(ZombieDetectorPipeline):
    _local_dir: Path = submission_models_dir() / "faster_rcnn"

    @property
    def pipeline_id(self) -> str:
        return "faster_rcnn"

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
                "Either copy an experiment directory here, or point the benchmark "
                "at the experiment folder with --checkpoint.",
            )
        self._detector = ZombieDetector(
            model_path=str(self._local_dir),
            model_type="faster_rcnn",
            config_path=config_path,
            device=device,
            conf_threshold=conf_threshold,
            resize="auto",
            preprocessing_variant="auto",
        )

    def detect(self, observation: np.ndarray) -> np.ndarray:
        return self._detector.detect(observation)

    def describe(self) -> str:
        return f"faster_rcnn device={self._detector.device} dir={self._local_dir}"

