"""Template matching — ``submission_models/template_match/`` (``template_gray.npy``).

Best mixed test precision on mixed test in the matrix run: **0.1263** (baseline;
use mainly for speed profiling, not accuracy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from zombie_detection.inference import ZombieDetector
from zombie_detection.realtime.deployment._paths import submission_models_dir
from zombie_detection.realtime.interface import ZombieDetectorPipeline


class TemplateMatchDeploymentPipeline(ZombieDetectorPipeline):
    mixed_test_precision: float = 0.1262626262626262
    _local_dir: Path = submission_models_dir() / "template_match"

    @property
    def pipeline_id(self) -> str:
        return "template_match"

    def __init__(
        self,
        *,
        device: str = "cpu",
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
            model_type="template_match",
            config_path=config_path,
            device="cpu",
            conf_threshold=conf_threshold,
            resize=None,
            preprocessing_variant="rgb",
        )

    def detect(self, observation: np.ndarray) -> np.ndarray:
        return self._detector.detect(observation)

    def describe(self) -> str:
        return f"template_match mixed_test_precision={self.mixed_test_precision:.4f} dir={self._local_dir}"
