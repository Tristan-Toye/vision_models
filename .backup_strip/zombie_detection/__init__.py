"""Zombie detection computer vision pipeline for KAZ (runtime + experiments)."""

from zombie_detection.interface import ZombieDetectorPipeline
from zombie_detection.yolov11n import YOLOv11nPipeline

__all__ = ["ZombieDetectorPipeline", "YOLOv11nPipeline"]
