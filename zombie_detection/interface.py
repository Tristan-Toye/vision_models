

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ZombieDetectorPipeline(ABC):
    

    @property
    @abstractmethod
    def pipeline_id(self) -> str:
        

    @abstractmethod
    def detect(self, observation: np.ndarray) -> np.ndarray:
        

    def warmup(self, observation: np.ndarray | None = None) -> None:
        
        h, w = 720, 1280
        dummy = observation
        if dummy is None:
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
        self.detect(dummy)

    def describe(self) -> str:
        
        return f"{self.pipeline_id} ({type(self).__name__})"

    def __call__(self, observation: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        return self.detect(observation)
