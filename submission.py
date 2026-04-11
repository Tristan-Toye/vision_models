"""Submission file for Task 3 (multi agent KAZ).
"""
import os
from typing import Callable

import numpy as np
import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


class CustomWrapper(BaseWrapper):
    """Pass-through wrapper (preprocessing is handled inside the detector)."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return super().observation_space(agent)

    def observe(self, agent: AgentID) -> ObsType | None:
        return super().observe(agent)


class CustomPredictFunction(Callable):
    """Placeholder for RL agent (not implemented in CV-only mode)."""

    def __init__(self, env: gymnasium.Env):
        self.env = env

    def __call__(self, observation, agent, *args, **kwargs):
        return self.env.action_space(agent).sample()


class CustomZombieDetectorFunction(Callable):
    """Detect zombies using the trained CV model.

    Returns a matrix of shape (nb_zombies, 4) where each row is
    (x, y, width, height) ordered from most confident to least.
    """

    def __init__(self, env: gymnasium.Env):
        from zombie_detection.inference import ZombieDetector

        model_path = os.path.join(PACKAGE_DIR, "zombie_detection", "checkpoints", "best_model.pt")
        config_path = os.path.join(PACKAGE_DIR, "zombie_detection", "config.yaml")

        # Default to yolov8n; change this after running experiments to the best model
        self.detector = ZombieDetector(
            model_path=model_path,
            model_type="yolov8n",
            config_path=config_path,
        )

    def __call__(self, observation, *args, **kwargs):
        return self.detector.detect(observation)

