"""Model registry for zombie detection experiments."""

from zombie_detection.models.heatmap_cnn import HeatmapCNN
from zombie_detection.models.resnet_backbone import ResNetDetector
from zombie_detection.models.yolo_wrapper import YOLODetector
from zombie_detection.models.fasterrcnn_wrapper import FasterRCNNDetector
from zombie_detection.models.detr_wrapper import RTDETRDetector

# NOTE: `sliding_window` depends on optional third-party packages (e.g. sklearn).
# Keep it lazily imported so environments without those deps can still run
# other models (submission pipelines, benchmarking, etc.).

MODEL_REGISTRY = {
    "heatmap_cnn": HeatmapCNN,
    "resnet18_head": lambda **kw: ResNetDetector(backbone="resnet18", **kw),
    "resnet50_head": lambda **kw: ResNetDetector(backbone="resnet50", **kw),
    "yolov8n": lambda **kw: YOLODetector(model_name="yolov8n", **kw),
    "yolov11n": lambda **kw: YOLODetector(model_name="yolo11n", **kw),
    "faster_rcnn": FasterRCNNDetector,
    "rt_detr": RTDETRDetector,
    "hog_svm": "zombie_detection.models.sliding_window:HOGSVMDetector",
    "template_match": "zombie_detection.models.sliding_window:TemplateMatchDetector",
}

# Models that use their own training loop (not the generic train.py loop)
SELF_TRAINING_MODELS = {"yolov8n", "yolov11n", "rt_detr", "hog_svm", "template_match"}

# Models that use heatmap targets vs bbox targets
HEATMAP_MODELS = {"heatmap_cnn", "resnet18_head", "resnet50_head"}
BBOX_MODELS = {"faster_rcnn"}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    factory = MODEL_REGISTRY[name]
    if isinstance(factory, str):
        # Lazy import to avoid importing optional deps unless needed.
        mod_path, attr = factory.split(":", 1)
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            factory = getattr(mod, attr)
        except Exception as e:
            raise ModuleNotFoundError(
                f"Model {name!r} requires optional dependencies. "
                f"Failed to import {factory} ({type(e).__name__}: {e})."
            ) from e
    return factory(**kwargs)


def get_target_mode(model_name: str) -> str:
    if model_name in HEATMAP_MODELS:
        return "heatmap"
    if model_name in BBOX_MODELS:
        return "bbox"
    return "bbox"
