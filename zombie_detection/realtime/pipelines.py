"""Concrete :class:`ZombieDetectorPipeline` implementations for live play.

Selected for a **mix of latency profiles** (CPU-friendly small nets vs
Ultralytics detectors). Pick 3–5 to benchmark on your target hardware.

Environment variables (see :func:`build_pipeline_from_env`):

* ``RLK_ZOMBIE_PIPELINE`` — pipeline id (default ``heatmap_cnn``).
* ``RLK_ZOMBIE_CHECKPOINT`` — file (``best_model.pt``, ``.../weights/best.pt``)
  **or** experiment directory containing weights + ``train_settings.json``.
* ``RLK_ZOMBIE_DEVICE`` — ``cpu`` or ``cuda`` (default ``cpu``).
* ``RLK_ZOMBIE_CONF`` — float confidence / heatmap threshold (default ``0.35``).
* ``RLK_ZOMBIE_CONFIG`` — optional path to ``config.yaml``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np

from zombie_detection.inference import ZombieDetector
from zombie_detection.realtime.interface import ZombieDetectorPipeline


def _checkpoint_from_env() -> str | None:
    return os.environ.get("RLK_ZOMBIE_CHECKPOINT")


def _device_from_env() -> str:
    return os.environ.get("RLK_ZOMBIE_DEVICE", "auto")


def _conf_from_env() -> float:
    return float(os.environ.get("RLK_ZOMBIE_CONF", "0.35"))


def _config_from_env() -> str | None:
    v = os.environ.get("RLK_ZOMBIE_CONFIG")
    return v if v else None


class _TorchCheckpointPipeline(ZombieDetectorPipeline):
    """Shared loader for models served by :class:`zombie_detection.inference.ZombieDetector`."""

    model_type: ClassVar[str]

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        config_path: str | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
    ):
        ck = checkpoint or _checkpoint_from_env()
        if not ck:
            raise ValueError(
                f"{self.model_type}: set RLK_ZOMBIE_CHECKPOINT to a checkpoint file "
                "or experiment directory (see zombie_detection/realtime/README.txt).",
            )
        self._checkpoint = str(Path(ck).expanduser().resolve())
        self._detector = ZombieDetector(
            model_path=self._checkpoint,
            model_type=self.model_type,
            config_path=config_path or _config_from_env(),
            device=device or _device_from_env(),
            conf_threshold=conf_threshold if conf_threshold is not None else _conf_from_env(),
            resize="auto",
            preprocessing_variant="auto",
        )

    @property
    def pipeline_id(self) -> str:
        return self.model_type

    def detect(self, observation: np.ndarray) -> np.ndarray:
        return self._detector.detect(observation)

    def describe(self) -> str:
        return f"{self.pipeline_id} checkpoint={self._checkpoint} device={self._detector.device}"


class HeatmapCNNPipeline(_TorchCheckpointPipeline):
    """Small U-Net heatmap model — good first choice for CPU real-time."""

    model_type = "heatmap_cnn"


class ResNet18HeadPipeline(_TorchCheckpointPipeline):
    """ResNet-18 + heatmap decoder — stronger than heatmap_cnn, still 360×640."""

    model_type = "resnet18_head"


class YOLOv8nPipeline(_TorchCheckpointPipeline):
    """Ultralytics YOLOv8n — smaller / often faster than v11."""

    model_type = "yolov8n"


class YOLOv11nPipeline(_TorchCheckpointPipeline):
    """Ultralytics YOLOv11n — typically best accuracy among YOLO variants here."""

    model_type = "yolov11n"


class RTDETRPipeline(_TorchCheckpointPipeline):
    """RT-DETR — transformer detector; heavier than YOLO on CPU."""

    model_type = "rt_detr"


class HogSvmPipeline(_TorchCheckpointPipeline):
    """HOG + linear SVM sliding window — CPU-only; can be slow per frame at native res."""

    model_type = "hog_svm"


class TemplateMatchPipeline(ZombieDetectorPipeline):
    """OpenCV template matching — very fast on CPU; accuracy depends on template."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        config_path: str | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
    ):
        ck = checkpoint or _checkpoint_from_env()
        if not ck:
            raise ValueError(
                "template_match: set RLK_ZOMBIE_CHECKPOINT to an experiment directory "
                "containing template_gray.npy (written at train time) or hog_svm.joblib parent.",
            )
        self._checkpoint = str(Path(ck).expanduser().resolve())
        self._detector = ZombieDetector(
            model_path=self._checkpoint,
            model_type="template_match",
            config_path=config_path or _config_from_env(),
            device="cpu",
            conf_threshold=conf_threshold if conf_threshold is not None else _conf_from_env(),
            resize=None,
            preprocessing_variant="rgb",
        )

    @property
    def pipeline_id(self) -> str:
        return "template_match"

    def detect(self, observation: np.ndarray) -> np.ndarray:
        return self._detector.detect(observation)

    def describe(self) -> str:
        return f"template_match dir={self._checkpoint}"


#: Registry of pipeline id -> class (instantiated with default env kwargs).
PIPELINE_REGISTRY: dict[str, type[ZombieDetectorPipeline]] = {
    "heatmap_cnn": HeatmapCNNPipeline,
    "resnet18_head": ResNet18HeadPipeline,
    "faster_rcnn": type(
        "FasterRCNNPipeline",
        (_TorchCheckpointPipeline,),
        {"model_type": "faster_rcnn", "__doc__": "TorchVision Faster R-CNN (two-stage)."},
    ),
    "yolov8n": YOLOv8nPipeline,
    "yolov11n": YOLOv11nPipeline,
    "rt_detr": RTDETRPipeline,
    "hog_svm": HogSvmPipeline,
    "template_match": TemplateMatchPipeline,
}


def create_pipeline(
    pipeline_id: str,
    checkpoint: str | Path | None = None,
    **kwargs,
) -> ZombieDetectorPipeline:
    """Instantiate a pipeline by id (see :data:`PIPELINE_REGISTRY`)."""
    key = pipeline_id.strip().lower()
    if key not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline_id={pipeline_id!r}. Choose one of: {sorted(PIPELINE_REGISTRY)}",
        )
    cls = PIPELINE_REGISTRY[key]
    return cls(checkpoint=checkpoint, **kwargs)


def build_pipeline_from_env() -> ZombieDetectorPipeline:
    """Build pipeline from ``RLK_ZOMBIE_PIPELINE`` and related env vars."""
    pid = os.environ.get("RLK_ZOMBIE_PIPELINE", "heatmap_cnn").strip().lower()
    return create_pipeline(pid)
