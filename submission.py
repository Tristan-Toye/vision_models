"""Submission file for Task 3 (multi agent KAZ).

Runtime is fixed to a single model for deployment simplicity:
**YOLOv11n** (Ultralytics) with weights shipped in-repo under
``zombie_detection/realtime/runtime_models/yolov11n/``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import gymnasium
import numpy as np
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

PACKAGE_DIR = str(Path(__file__).resolve().parent)
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)


class CustomWrapper(BaseWrapper):
    """Pass-through wrapper (observations stay default RGB HWC unless you change this)."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return self.env.observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.env.observe(agent)


class CustomPredictFunction(Callable):
    """Placeholder for RL agent (not implemented in CV-only mode)."""

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, agent, *args, **kwargs):
        pass


class CustomZombieDetectorFunction(Callable):
    """Detect zombies using the fixed runtime pipeline (YOLOv11n).

    Returns ``(N, 4)`` float32 ``[x, y, width, height]`` in screen pixels, sorted
    by confidence (evaluation contract).
    """

    def __init__(self, env: gymnasium.Env):
        from zombie_detection.yolov11n import YOLOv11nPipeline

        self._env = env
        self._pipeline = YOLOv11nPipeline(device="auto", conf_threshold=0.35)

    def __call__(self, observation, *args, **kwargs):
        """Return ``(N,4)`` float32 ``[x,y,w,h]``; empty array if no detections."""
        if observation is None:
            return np.zeros((0, 4), dtype=np.float32)

        obs = np.asarray(observation)
        if obs.ndim != 3 or obs.shape[2] != 3:
            raise ValueError(
                f"CustomZombieDetectorFunction expected (H,W,3) uint8 RGB; got shape {obs.shape}. "
                "Adjust CustomWrapper if your agent uses flattened or normalized observations.",
            )
        if obs.dtype != np.uint8:
            obs = np.clip(obs, 0, 255).astype(np.uint8)

        return self._pipeline.detect(obs)
