"""Submission-oriented detectors: fixed paths under ``submission_models/``."""

from __future__ import annotations

from pathlib import Path
from typing import Type

import yaml

from zombie_detection.realtime.deployment.heatmap_cnn import HeatmapCNNDeploymentPipeline
from zombie_detection.realtime.deployment.faster_rcnn import FasterRCNNDeploymentPipeline
from zombie_detection.realtime.deployment.resnet18_head import ResNet18HeadDeploymentPipeline
from zombie_detection.realtime.deployment.rt_detr import RTDETRDeploymentPipeline
from zombie_detection.realtime.deployment.template_match import TemplateMatchDeploymentPipeline
from zombie_detection.realtime.deployment.yolov11n import YOLOv11nDeploymentPipeline
from zombie_detection.realtime.deployment.yolov8n import YOLOv8nDeploymentPipeline
from zombie_detection.realtime.interface import ZombieDetectorPipeline

# Registry key = submission_config.yaml ``active_model`` value (best mixed accuracy first in docs).
SUBMISSION_MODEL_REGISTRY: dict[str, Type[ZombieDetectorPipeline]] = {
    "rt_detr": RTDETRDeploymentPipeline,
    "yolov11n": YOLOv11nDeploymentPipeline,
    "heatmap_cnn": HeatmapCNNDeploymentPipeline,
    "resnet18_head": ResNet18HeadDeploymentPipeline,
    "yolov8n": YOLOv8nDeploymentPipeline,
    "template_match": TemplateMatchDeploymentPipeline,
    "faster_rcnn": FasterRCNNDeploymentPipeline,
}


def load_pipeline_from_submission_config(config_path: str | Path) -> ZombieDetectorPipeline:
    """Instantiate the pipeline declared in ``submission_config.yaml``."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Submission config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    key = str(data.get("active_model", "heatmap_cnn")).strip().lower()
    if key not in SUBMISSION_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown active_model={key!r}. Choose one of: {sorted(SUBMISSION_MODEL_REGISTRY)}",
        )
    cls = SUBMISSION_MODEL_REGISTRY[key]
    device_raw = data.get("device", "auto")
    device = str(device_raw).strip() if device_raw is not None else "auto"
    conf = float(data.get("conf_threshold", 0.35))
    cpp = data.get("config_path")
    config_path_kw: str | None = cpp if isinstance(cpp, str) and cpp.strip() else None
    return cls(device=device, conf_threshold=conf, config_path=config_path_kw)


__all__ = [
    "SUBMISSION_MODEL_REGISTRY",
    "load_pipeline_from_submission_config",
    "RTDETRDeploymentPipeline",
    "YOLOv11nDeploymentPipeline",
    "HeatmapCNNDeploymentPipeline",
    "ResNet18HeadDeploymentPipeline",
    "YOLOv8nDeploymentPipeline",
    "TemplateMatchDeploymentPipeline",
    "FasterRCNNDeploymentPipeline",
]
