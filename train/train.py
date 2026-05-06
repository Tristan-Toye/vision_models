"""Training framework for zombie detection models.

Supports:
- Heatmap-based models (HeatmapCNN, ResNet+head) with generic PyTorch loop
- Faster R-CNN with its own loss computation
- YOLO / RT-DETR via Ultralytics API (self-training)
- HOG+SVM / template matching (classical, no gradient loop)

Produces loss curves (train + val) and precision curves over epochs.

Usage:
    python -m train.train --model heatmap_cnn --config zombie_detection/config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.dataset import (
    ZombieDetectionDataset,
    bbox_collate_fn,
    heatmap_collate_fn,
)
from train.preprocessing import build_preprocessing_pipeline
from train.models import (
    get_model,
    get_target_mode,
    SELF_TRAINING_MODELS,
    HEATMAP_MODELS,
)
from train.models.heatmap_cnn import heatmap_to_boxes


# ──────────────────────────── Loss functions ────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for heatmap regression to handle class imbalance."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pos_mask = target.ge(0.99).float()
        neg_mask = target.lt(0.99).float()

        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_loss = -((1 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred) * neg_mask

        n_pos = pos_mask.sum().clamp(min=1)
        loss = (pos_loss.sum() + neg_loss.sum()) / n_pos
        return loss


def get_loss_fn(name: str) -> nn.Module:
    """Return a loss function by name."""
    losses = {
        "mse": nn.MSELoss(),
        "focal": FocalLoss(),
        "bce": nn.BCELoss(),
        "smooth_l1": nn.SmoothL1Loss(),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    return losses[name]


# ──────────────────────────── Evaluation ────────────────────────────

def compute_precision_iou(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    """Compute detection precision using IoU >= threshold greedy matching.
    Same metric as evaluation.py."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0

    matched = np.zeros(len(gt_boxes), dtype=bool)
    found = 0

    for pred in pred_boxes:
        for gi, gt in enumerate(gt_boxes):
            if matched[gi]:
                continue
            if _iou(pred, gt) >= iou_threshold:
                found += 1
                matched[gi] = True
                break

    return found / len(gt_boxes)


def _iou(r1, r2):
    l1, t1, w1, h1 = r1
    r1r = l1 + w1
    b1 = t1 + h1
    l2, t2, w2, h2 = r2
    r2r = l2 + w2
    b2 = t2 + h2
    xl = max(l1, l2)
    yt = max(t1, t2)
    xr = min(r1r, r2r)
    yb = min(b1, b2)
    if xr < xl or yb < yt:
        return 0.0
    inter = (xr - xl) * (yb - yt)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter / float(area1 + area2 - inter)


# ──────────────────────────── Training loops ────────────────────────────

def train_heatmap_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    cfg: dict,
    output_dir: Path,
    model_name: str,
) -> dict:
    """Generic training loop for heatmap-based models."""
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    epochs = cfg["training"]["default_epochs"]
    lr = cfg["training"]["default_lr"]
    bbox_w = cfg["zombie"]["bbox_width"]
    bbox_h = cfg["zombie"]["bbox_height"]
    screen_w = cfg["screen"]["width"]
    screen_h = cfg["screen"]["height"]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_precision": [],
    }
    best_precision = 0.0
    best_epoch = 0

    epoch_bar = tqdm(range(epochs), desc=f"[{model_name}] Epochs", unit="ep")
    for epoch in epoch_bar:
        # ── Train ──
        model.train()
        train_losses = []
        train_bar = tqdm(
            train_loader, desc=f"  Train ep {epoch+1}", leave=False, unit="batch",
        )
        for images, targets in train_bar:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = np.mean(train_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        precisions = []

        val_bar = tqdm(
            val_loader, desc=f"  Val   ep {epoch+1}", leave=False, unit="batch",
        )
        with torch.no_grad():
            for images, targets in val_bar:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss = loss_fn(preds, targets)
                val_losses.append(loss.item())

                for i in range(preds.shape[0]):
                    hm = preds[i, 0].cpu().numpy()
                    pred_boxes = heatmap_to_boxes(hm, bbox_w, bbox_h, screen_w, screen_h)

                    gt_hm = targets[i, 0].cpu().numpy()
                    gt_boxes = heatmap_to_boxes(gt_hm, bbox_w, bbox_h, screen_w, screen_h, threshold=0.5)

                    precisions.append(compute_precision_iou(pred_boxes, gt_boxes))

        avg_val_loss = np.mean(val_losses)
        avg_precision = np.mean(precisions) if precisions else 0.0

        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_precision"].append(avg_precision)

        star = " *" if avg_precision > best_precision else ""
        epoch_bar.set_postfix(
            t_loss=f"{avg_train_loss:.4f}",
            v_loss=f"{avg_val_loss:.4f}",
            v_prec=f"{avg_precision:.4f}{star}",
        )

        if avg_precision > best_precision:
            best_precision = avg_precision
            best_epoch = epoch
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))

    torch.save(model.state_dict(), str(output_dir / "last_model.pt"))

    history["best_epoch"] = best_epoch
    history["best_precision"] = best_precision

    _plot_curves(history, output_dir, model_name)
    _save_history(history, output_dir, model_name)
    return history


def train_fasterrcnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    output_dir: Path,
    model_name: str,
) -> dict:
    """Training loop for Faster R-CNN which computes its own loss."""
    device = cfg["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    epochs = cfg["training"]["default_epochs"]
    lr = cfg["training"]["default_lr"]
    bbox_w = cfg["zombie"]["bbox_width"]
    bbox_h = cfg["zombie"]["bbox_height"]

    model = model.to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_precision": []}
    best_precision = 0.0
    best_epoch = 0

    epoch_bar = tqdm(range(epochs), desc=f"[{model_name}] Epochs", unit="ep")
    for epoch in epoch_bar:
        # ── Train ──
        model.train()
        train_losses = []

        train_bar = tqdm(
            train_loader, desc=f"  Train ep {epoch+1}", leave=False, unit="batch",
        )
        for images, targets in train_bar:
            images = [img.to(device) for img in images.unbind(0)]
            targets_dev = []
            for t in targets:
                targets_dev.append({
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device),
                })

            loss_dict = model(images, targets_dev)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            train_losses.append(losses.item())
            train_bar.set_postfix(loss=f"{losses.item():.4f}")

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # ── Validate (loss) ──
        model.train()
        val_losses = []
        val_loss_bar = tqdm(
            val_loader, desc=f"  Val-L ep {epoch+1}", leave=False, unit="batch",
        )
        with torch.no_grad():
            for images, targets in val_loss_bar:
                imgs = [img.to(device) for img in images.unbind(0)]
                targets_dev = []
                for t in targets:
                    targets_dev.append({
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device),
                    })
                loss_dict = model(imgs, targets_dev)
                val_losses.append(sum(loss_dict.values()).item())

        avg_val_loss = np.mean(val_losses)

        # ── Validate (precision) ──
        model.eval()
        precisions = []

        val_prec_bar = tqdm(
            val_loader, desc=f"  Val-P ep {epoch+1}", leave=False, unit="batch",
        )
        with torch.no_grad():
            for images, targets in val_prec_bar:
                imgs = [img.to(device) for img in images.unbind(0)]
                outputs = model(imgs)

                for pred, gt in zip(outputs, targets):
                    pred_boxes = pred["boxes"].cpu().numpy()
                    scores = pred["scores"].cpu().numpy()
                    keep = scores >= 0.5
                    pred_xywh = np.zeros((keep.sum(), 4))
                    if keep.sum() > 0:
                        pb = pred_boxes[keep]
                        pred_xywh[:, 0] = pb[:, 0]
                        pred_xywh[:, 1] = pb[:, 1]
                        pred_xywh[:, 2] = pb[:, 2] - pb[:, 0]
                        pred_xywh[:, 3] = pb[:, 3] - pb[:, 1]

                    gt_xyxy = gt["boxes"].numpy()
                    gt_xywh = np.zeros_like(gt_xyxy)
                    if len(gt_xyxy) > 0:
                        gt_xywh[:, 0] = gt_xyxy[:, 0]
                        gt_xywh[:, 1] = gt_xyxy[:, 1]
                        gt_xywh[:, 2] = gt_xyxy[:, 2] - gt_xyxy[:, 0]
                        gt_xywh[:, 3] = gt_xyxy[:, 3] - gt_xyxy[:, 1]

                    precisions.append(compute_precision_iou(pred_xywh, gt_xywh))

        avg_precision = np.mean(precisions) if precisions else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_precision"].append(avg_precision)

        star = " *" if avg_precision > best_precision else ""
        epoch_bar.set_postfix(
            t_loss=f"{avg_train_loss:.4f}",
            v_loss=f"{avg_val_loss:.4f}",
            v_prec=f"{avg_precision:.4f}{star}",
        )

        if avg_precision > best_precision:
            best_precision = avg_precision
            best_epoch = epoch
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))

    torch.save(model.state_dict(), str(output_dir / "last_model.pt"))
    history["best_epoch"] = best_epoch
    history["best_precision"] = best_precision
    _plot_curves(history, output_dir, model_name)
    _save_history(history, output_dir, model_name)
    return history


def train_yolo_or_detr(
    model_name: str,
    cfg: dict,
    output_dir: Path,
) -> dict:
    """Train YOLO or RT-DETR using Ultralytics API."""
    from train.models import get_model

    detector = get_model(model_name)
    ds_cfg = cfg["dataset"]
    data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]

    # Export to YOLO format
    # Shared cache across YOLOv8/YOLOv11/RT-DETR experiments to avoid duplicating disk usage.
    # Store alongside the experiment results root (e.g. USB) so it doesn't fill the SSD.
    results_root = output_dir.parent
    yolo_dir = results_root / "_shared_yolo_dataset" / ds_cfg["name"]
    resize_hw = cfg.get("preprocessing", {}).get("resize")
    if hasattr(detector, "export_dataset_to_yolo_format"):
        yaml_path = detector.export_dataset_to_yolo_format(
            data_dir, yolo_dir,
            screen_w=cfg["screen"]["width"],
            screen_h=cfg["screen"]["height"],
            image_format="png",
            resize_hw=tuple(resize_hw) if resize_hw is not None else (360, 640),
        )
    else:
        # RT-DETR uses same YOLO format
        from train.models.yolo_wrapper import YOLODetector
        exporter = YOLODetector()
        yaml_path = exporter.export_dataset_to_yolo_format(
            data_dir, yolo_dir,
            screen_w=cfg["screen"]["width"],
            screen_h=cfg["screen"]["height"],
            image_format="png",
            resize_hw=tuple(resize_hw) if resize_hw is not None else (360, 640),
        )

    results = detector.train(
        dataset_yaml=yaml_path,
        epochs=cfg["training"]["default_epochs"],
        batch_size=cfg["training"]["default_batch_size"],
        project=str(output_dir),
        name=model_name,
    )

    return {"model": model_name, "results": str(results)}


def train_classical(
    model_name: str,
    cfg: dict,
    output_dir: Path,
) -> dict:
    """Train classical detectors (HOG+SVM, template matching)."""
    from train.models import get_model

    detector = get_model(model_name)
    ds_cfg = cfg["dataset"]
    train_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"] / "train"

    bbox_w = cfg["zombie"]["bbox_width"]
    bbox_h = cfg["zombie"]["bbox_height"]

    if model_name == "hog_svm":
        detector.train_from_dataset(train_dir, bbox_w=bbox_w, bbox_h=bbox_h)
        detector.save(str(output_dir / "hog_svm.joblib"))
    elif model_name == "template_match":
        detector.extract_template_from_dataset(train_dir, bbox_w=bbox_w, bbox_h=bbox_h)
        if detector.template is not None:
            np.save(str(output_dir / "template_gray.npy"), detector.template)

    # Evaluate on val set
    val_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"] / "val"
    precisions = _evaluate_classical(detector, val_dir)
    avg_p = np.mean(precisions) if precisions else 0.0

    print(f"[{model_name}] Val precision: {avg_p:.4f}")
    return {"model": model_name, "val_precision": avg_p}


def _evaluate_classical(detector, data_dir: Path) -> list:
    obs_files = sorted(data_dir.glob("*_obs.npy"))
    precisions = []
    for obs_path in tqdm(obs_files, desc="  Classical val", leave=False, unit="img"):
        stem = obs_path.name.replace("_obs.npy", "")
        box_path = obs_path.parent / f"{stem}_zombies.npy"
        if not box_path.exists():
            continue
        image = np.load(str(obs_path))
        gt_boxes = np.load(str(box_path))
        pred_boxes = detector.predict(image)
        precisions.append(compute_precision_iou(pred_boxes, gt_boxes))
    return precisions


# ──────────────────────────── Plotting ────────────────────────────

def _plot_curves(history: dict, output_dir: Path, model_name: str):
    """Save loss and precision curves as PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model_name} - Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_precision"], label="Val Precision", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Precision (IoU >= 0.5)")
    ax2.set_title(f"{model_name} - Validation Precision")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / f"{model_name}_curves.png"), dpi=150)
    plt.close(fig)


def _save_history(history: dict, output_dir: Path, model_name: str):
    """Persist raw training history as JSON for later analysis."""
    history_path = output_dir / f"{model_name}_history.json"
    serialisable = {}
    for k, v in history.items():
        if isinstance(v, (list, tuple)):
            serialisable[k] = [float(x) if isinstance(x, (float, np.floating)) else x for x in v]
        elif isinstance(v, (float, np.floating)):
            serialisable[k] = float(v)
        else:
            serialisable[k] = v
    with open(history_path, "w") as f:
        json.dump(serialisable, f, indent=2)


# ──────────────────────────── Main entry ────────────────────────────

def train_model(
    model_name: str,
    cfg: dict,
    output_dir: Optional[Path] = None,
    loss_name: str = "mse",
    preprocessing_variant: str = "rgb",
    pretrained_mode: str = "finetune",
    resize: "list[int] | None" = "from_config",
    manual_features: bool = False,
) -> dict:
    """Unified entry point: dispatches to the right training loop.

    Args:
        resize: [h, w] to resize images, None to keep native resolution,
                or "from_config" to read from cfg["preprocessing"]["resize"].
        manual_features: whether to append handcrafted feature channels.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "zombie_detection" / "checkpoints" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve resize setting
    if resize == "from_config":
        resize = cfg["preprocessing"].get("resize")

    # Save settings used for this experiment so inference can match
    with open(output_dir / "train_settings.json", "w") as f:
        json.dump({
            "resize": resize,
            "preprocessing_variant": preprocessing_variant,
            "manual_features": manual_features,
        }, f)

    # Self-training models (YOLO, RT-DETR, classical)
    if model_name in {"yolov8n", "yolov11n", "rt_detr"}:
        return train_yolo_or_detr(model_name, cfg, output_dir)

    if model_name in {"hog_svm", "template_match"}:
        return train_classical(model_name, cfg, output_dir)

    # PyTorch-based models
    ds_cfg = cfg["dataset"]
    data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]

    target_mode = get_target_mode(model_name)
    augment = True
    transform = build_preprocessing_pipeline(
        cfg, variant=preprocessing_variant, augment=augment,
        resize=resize, manual_features=manual_features,
    )
    val_transform = build_preprocessing_pipeline(
        cfg, variant=preprocessing_variant, augment=False,
        resize=resize, manual_features=manual_features,
    )

    # Heatmap size matches actual image dimensions after preprocessing
    if resize is not None:
        hm_size = tuple(resize)
    else:
        hm_size = (cfg["screen"]["height"], cfg["screen"]["width"])

    train_ds = ZombieDetectionDataset(
        data_dir / "train", target_mode=target_mode, transform=transform,
        heatmap_size=hm_size,
    )
    val_ds = ZombieDetectionDataset(
        data_dir / "val", target_mode=target_mode, transform=val_transform,
        heatmap_size=hm_size,
    )

    collate = heatmap_collate_fn if target_mode == "heatmap" else bbox_collate_fn
    batch_size = cfg["training"]["default_batch_size"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    # Build model
    from train.preprocessing import NUM_MANUAL_FEATURES
    in_channels = 4 if preprocessing_variant == "rgb_edge" else 3
    if manual_features:
        in_channels += NUM_MANUAL_FEATURES
    model_kwargs = {"in_channels": in_channels}

    if model_name in {"resnet18_head", "resnet50_head"}:
        model_kwargs["pretrained"] = pretrained_mode != "scratch"
        model_kwargs["freeze_backbone"] = pretrained_mode == "frozen"

    if model_name == "faster_rcnn":
        model_kwargs["pretrained"] = pretrained_mode != "scratch"
        model_kwargs["freeze_backbone"] = pretrained_mode == "frozen"
        model = get_model(model_name, **model_kwargs)
        return train_fasterrcnn(model, train_loader, val_loader, cfg, output_dir, model_name)

    model = get_model(model_name, **model_kwargs)

    # Heatmap models
    loss_fn = get_loss_fn(loss_name)
    return train_heatmap_model(model, train_loader, val_loader, loss_fn, cfg, output_dir, model_name)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train a zombie detection model")
    parser.add_argument("--model", required=True, help="Model name from registry")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "zombie_detection" / "config.yaml"))
    parser.add_argument("--loss", default="mse", help="Loss function name")
    parser.add_argument("--preprocessing", default="rgb", help="Preprocessing variant")
    parser.add_argument("--pretrained", default="finetune", choices=["frozen", "finetune", "scratch"])
    parser.add_argument("--resize", default="from_config", help="'from_config', 'none', or 'H,W' e.g. '360,640'")
    parser.add_argument("--output-dir", default=None, help="Output directory for checkpoints")
    parser.add_argument("--manual-features", action="store_true", help="Append handcrafted feature channels")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.resize == "none":
        resize_val = None
    elif args.resize == "from_config":
        resize_val = "from_config"
    else:
        resize_val = [int(x) for x in args.resize.split(",")]

    out = Path(args.output_dir) if args.output_dir else None
    results = train_model(
        model_name=args.model,
        cfg=cfg,
        output_dir=out,
        resize=resize_val,
        loss_name=args.loss,
        preprocessing_variant=args.preprocessing,
        pretrained_mode=args.pretrained,
        manual_features=args.manual_features,
    )

    results_path = (out or PROJECT_ROOT / "zombie_detection" / "checkpoints" / args.model) / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
