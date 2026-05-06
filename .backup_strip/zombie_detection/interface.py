"""Abstract interface for real-time zombie bounding-box detection.

Implementations wrap trained checkpoints and expose a single ``detect`` entry
point compatible with :class:`submission.CustomZombieDetectorFunction`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ZombieDetectorPipeline(ABC):
    """Pluggable zombie detector for live KAZ / PettingZoo use.

    Contract:
        * ``observation`` is ``(H, W, 3)`` uint8 RGB in **screen pixel** coordinates
          (same frame format as the simulator / dataset).
        * Return shape ``(N, 4)`` ``float32``, each row ``[x, y, w, h]`` in the same
          coordinate system, sorted from highest confidence to lowest.
    """

    @property
    @abstractmethod
    def pipeline_id(self) -> str:
        """Stable short name (e.g. ``heatmap_cnn``) for CLI / env switching."""

    @abstractmethod
    def detect(self, observation: np.ndarray) -> np.ndarray:
        """Run detection on one frame."""

    def warmup(self, observation: np.ndarray | None = None) -> None:
        """Optional one-frame JIT / cache warmup (default: one forward)."""
        h, w = 720, 1280
        dummy = observation
        if dummy is None:
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self.detect(dummy)

    def describe(self) -> str:
        """Human-readable one-line summary for logging."""
        return f"{self.pipeline_id} ({type(self).__name__})"

    def __call__(self, observation: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.detect(observation)
