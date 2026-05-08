"""Real-time zombie detector pipelines for live KAZ / submission integration."""

from zombie_detection.realtime.deployment import (
    SUBMISSION_MODEL_REGISTRY,
    load_pipeline_from_submission_config,
)
from zombie_detection.realtime.interface import ZombieDetectorPipeline
from zombie_detection.realtime.pipelines import (
    PIPELINE_REGISTRY,
    RTDETRPipeline,
    HeatmapCNNPipeline,
    ResNet18HeadPipeline,
    YOLOv8nPipeline,
    YOLOv11nPipeline,
    HogSvmPipeline,
    TemplateMatchPipeline,
    build_pipeline_from_env,
    create_pipeline,
)

__all__ = [
    "ZombieDetectorPipeline",
    "SUBMISSION_MODEL_REGISTRY",
    "load_pipeline_from_submission_config",
    "PIPELINE_REGISTRY",
    "HeatmapCNNPipeline",
    "ResNet18HeadPipeline",
    "YOLOv8nPipeline",
    "YOLOv11nPipeline",
    "RTDETRPipeline",
    "HogSvmPipeline",
    "TemplateMatchPipeline",
    "create_pipeline",
    "build_pipeline_from_env",
]
