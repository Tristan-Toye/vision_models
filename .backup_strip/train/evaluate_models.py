"""Experiment pipeline: train and evaluate all model configurations.

Iterates over the experiment matrix defined in config.yaml, trains each
model variant, evaluates on the test set per distortion level, and
produces comparison tables and charts.

Usage:
    python -m train.evaluate_models [--config zombie_detection/config.yaml]
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.train import train_model, compute_precision_iou, load_config
from train.models import SELF_TRAINING_MODELS, HEATMAP_MODELS, get_model, get_target_mode
from train.models.heatmap_cnn import heatmap_to_boxes
from train.preprocessing import build_preprocessing_pipeline
from train.dataset import ZombieDetectionDataset


# ───────── Which (model, loss, pretrained) combos are valid ─────────

# Loss functions only apply to heatmap-based PyTorch models
LOSS_APPLICABLE = {
    "heatmap_cnn": ["mse", "focal", "bce", "smooth_l1"],
    "resnet18_head": ["mse", "focal", "bce", "smooth_l1"],
    "resnet50_head": ["mse", "focal", "bce", "smooth_l1"],
    "faster_rcnn": ["default"],  # uses its own built-in loss
    "yolov8n": ["default"],
    "yolov11n": ["default"],
    "rt_detr": ["default"],
    "hog_svm": ["default"],
    "template_match": ["default"],
}

PRETRAINED_APPLICABLE = {
    "heatmap_cnn": ["scratch"],
    "resnet18_head": ["frozen", "finetune", "scratch"],
    "resnet50_head": ["frozen", "finetune", "scratch"],
    "faster_rcnn": ["frozen", "finetune", "scratch"],
    "yolov8n": ["finetune"],
    "yolov11n": ["finetune"],
    "rt_detr": ["finetune"],
    "hog_svm": ["scratch"],
    "template_match": ["scratch"],
}

MANUAL_FEATURES_APPLICABLE = {
    "heatmap_cnn": [True, False],
    "resnet18_head": [True, False],
    "resnet50_head": [True, False],
}


def _resize_label(resize_val) -> str:
    """Human-readable label for a resize setting."""
    if resize_val is None:
        return "native"
    return f"{resize_val[0]}x{resize_val[1]}"


def generate_experiment_configs(cfg: dict) -> list[dict]:
    """Generate all valid experiment configurations."""
    experiments = []
    exp_cfg = cfg["experiments"]

    resize_variants = exp_cfg.get("resize_variants", [cfg["preprocessing"].get("resize")])
    cfg_mf_modes = exp_cfg.get("manual_features_modes", [False])

    for model_name in exp_cfg["models"]:
        losses = LOSS_APPLICABLE.get(model_name, ["default"])
        pretrained_modes = PRETRAINED_APPLICABLE.get(model_name, ["scratch"])
        preproc_variants = exp_cfg.get("preprocessing_variants", ["rgb"])

        # Self-training models handle resize internally
        if model_name in SELF_TRAINING_MODELS:
            preproc_variants = ["rgb"]
            model_resize_variants = [None]  # YOLO/DETR handle their own resizing
        else:
            model_resize_variants = resize_variants

        applicable_losses = [l for l in losses if l in exp_cfg.get("loss_functions", ["mse"]) or l == "default"]
        applicable_pretrained = [p for p in pretrained_modes if p in exp_cfg.get("pretrained_modes", ["finetune"])]
        if not applicable_pretrained:
            # No config-listed mode is valid for this model — fall back to its first supported mode
            applicable_pretrained = [pretrained_modes[0]]

        model_mf_options = MANUAL_FEATURES_APPLICABLE.get(model_name, [False])
        applicable_mf = [m for m in model_mf_options if m in cfg_mf_modes]
        if not applicable_mf:
            applicable_mf = [False]

        for loss, pretrained, preproc, resize, mf in product(
            applicable_losses, applicable_pretrained, preproc_variants,
            model_resize_variants, applicable_mf,
        ):
            resize_lbl = _resize_label(resize)
            mf_lbl = "mf" if mf else "nomf"
            exp_id = f"{model_name}__{loss}__{pretrained}__{preproc}__{resize_lbl}__{mf_lbl}"
            experiments.append({
                "id": exp_id,
                "model": model_name,
                "loss": loss,
                "pretrained": pretrained,
                "preprocessing": preproc,
                "resize": resize,
                "manual_features": mf,
            })

    return experiments


def evaluate_on_test(
    model_name: str,
    checkpoint_dir: Path,
    cfg: dict,
    preprocessing_variant: str = "rgb",
    pretrained_mode: str = "finetune",
    resize: "list[int] | None" = None,
    manual_features: bool = False,
) -> dict:
    """Evaluate both best and last checkpoints on the test set.

    Returns a dict with 'best' and 'last' keys, each containing
    per-distortion-group precision results.
    """
    results = {}
    for ckpt_tag in ("best", "last"):
        ckpt_results = _evaluate_checkpoint(
            model_name, checkpoint_dir, cfg,
            preprocessing_variant, pretrained_mode, resize,
            checkpoint_name=f"{ckpt_tag}_model.pt",
            manual_features=manual_features,
        )
        results[ckpt_tag] = ckpt_results
    return results


def _evaluate_checkpoint(
    model_name: str,
    checkpoint_dir: Path,
    cfg: dict,
    preprocessing_variant: str,
    pretrained_mode: str,
    resize: "list[int] | None",
    checkpoint_name: str = "best_model.pt",
    manual_features: bool = False,
    manual_features_mask: list[bool] | None = None,
) -> dict:
    """Evaluate a single checkpoint on the test set, grouped by distortion level."""
    ds_cfg = cfg["dataset"]
    data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]
    test_dir = data_dir / "test"

    if not test_dir.exists():
        return {"error": "test directory not found"}

    ckpt_path = checkpoint_dir / checkpoint_name
    if not ckpt_path.exists():
        # YOLO/DETR may store weights elsewhere
        alt_paths = list(checkpoint_dir.glob("*/weights/best.pt"))
        if checkpoint_name == "best_model.pt" and alt_paths:
            ckpt_path = alt_paths[0]
        elif checkpoint_name == "last_model.pt":
            alt_paths = list(checkpoint_dir.glob("*/weights/last.pt"))
            if alt_paths:
                ckpt_path = alt_paths[0]
            else:
                return {"error": f"{checkpoint_name} not found"}

    # Pre-scan test files once (avoid 5x full scans for distortion groups)
    obs_files_all = sorted(test_dir.glob("*_obs.npy"))
    parsed = []
    for obs_path in obs_files_all:
        stem = obs_path.name.replace("_obs.npy", "")
        parts = stem.split("_d")
        if len(parts) < 2:
            continue
        try:
            file_level = int(parts[-1])
        except ValueError:
            continue
        box_path = test_dir / f"{stem}_zombies.npy"
        if not box_path.exists():
            continue
        parsed.append((file_level, obs_path, box_path))

    # For Ultralytics models, load detector once per checkpoint
    detector = None
    if model_name in SELF_TRAINING_MODELS:
        detector = get_model(model_name, checkpoint=str(ckpt_path))

    distortion_results = {}
    for group in cfg["experiments"]["distortion_eval_groups"]:
        group_name = group["name"]
        group_levels = group["levels"]

        precisions = []
        for file_level, obs_path, box_path in tqdm(
            parsed,
            desc=f"    Test [{checkpoint_name.replace('_model.pt','')}] {group_name}",
            leave=False, unit="img",
        ):
            if file_level not in group_levels:
                continue

            image = np.load(str(obs_path))
            gt_boxes = np.load(str(box_path))

            if model_name in SELF_TRAINING_MODELS:
                # Reuse loaded model. Only Ultralytics models accept predict kwargs like max_det.
                if model_name in {"yolov8n", "yolov11n", "rt_detr"}:
                    pred_boxes = detector.predict(image, max_det=100)
                else:
                    pred_boxes = detector.predict(image)
            elif model_name in HEATMAP_MODELS:
                pred_boxes = _predict_heatmap_model(
                    model_name, checkpoint_dir, image, cfg,
                    preprocessing_variant, pretrained_mode, resize,
                    checkpoint_name=checkpoint_name,
                    manual_features=manual_features,
                    manual_features_mask=manual_features_mask,
                )
            elif model_name == "faster_rcnn":
                pred_boxes = _predict_fasterrcnn(
                    checkpoint_dir, image, cfg, pretrained_mode, resize,
                    preprocessing_variant=preprocessing_variant,
                    checkpoint_name=checkpoint_name,
                )
            else:
                continue

            precisions.append(compute_precision_iou(pred_boxes, gt_boxes))

        distortion_results[group_name] = {
            "avg_precision": np.mean(precisions) if precisions else 0.0,
            "num_images": len(precisions),
        }

    return distortion_results


def _predict_heatmap_model(
    model_name, checkpoint_dir, image, cfg, preproc_variant, pretrained_mode, resize,
    checkpoint_name="best_model.pt",
    manual_features=False,
    manual_features_mask=None,
):
    import torch
    from train.preprocessing import NUM_MANUAL_FEATURES

    in_channels = 4 if preproc_variant == "rgb_edge" else 3
    if manual_features:
        in_channels += NUM_MANUAL_FEATURES
    model_kwargs = {"in_channels": in_channels}
    if model_name in {"resnet18_head", "resnet50_head"}:
        model_kwargs["pretrained"] = pretrained_mode != "scratch"
        model_kwargs["freeze_backbone"] = False

    model = get_model(model_name, **model_kwargs)
    ckpt = checkpoint_dir / checkpoint_name
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    transform = build_preprocessing_pipeline(
        cfg, variant=preproc_variant, augment=False, resize=resize,
        manual_features=manual_features,
        manual_features_mask=manual_features_mask,
    )
    boxes_dummy = np.zeros((0, 4), dtype=np.float32)
    image_proc, _ = transform(image, boxes_dummy)

    img_tensor = torch.from_numpy(image_proc).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        heatmap = model(img_tensor)[0, 0].cpu().numpy()

    bbox_w = cfg["zombie"]["bbox_width"]
    bbox_h = cfg["zombie"]["bbox_height"]
    screen_w = cfg["screen"]["width"]
    screen_h = cfg["screen"]["height"]

    return heatmap_to_boxes(heatmap, bbox_w, bbox_h, screen_w, screen_h)


def _predict_fasterrcnn(
    checkpoint_dir, image, cfg, pretrained_mode, resize,
    preprocessing_variant: str = "rgb",
    checkpoint_name="best_model.pt",
):
    import torch

    transform = build_preprocessing_pipeline(
        cfg, variant=preprocessing_variant, augment=False, resize=resize,
    )
    boxes_dummy = np.zeros((0, 4), dtype=np.float32)
    image_proc, _ = transform(image, boxes_dummy)

    in_ch = 4 if preprocessing_variant == "rgb_edge" else 3
    model = get_model(
        "faster_rcnn",
        pretrained=pretrained_mode != "scratch",
        freeze_backbone=False,
        in_channels=in_ch,
    )
    ckpt = checkpoint_dir / checkpoint_name
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    return model.predict(image_proc)


def create_comparison_table(all_results: list[dict]) -> pd.DataFrame:
    """Create a comparison DataFrame from experiment results.

    Test results now have a 'best' and 'last' sub-key.  We produce columns
    like  precision_best_mixed, precision_last_mixed, etc.
    """
    rows = []
    for r in all_results:
        row = {
            "experiment_id": r["id"],
            "model": r["model"],
            "loss": r["loss"],
            "pretrained": r["pretrained"],
            "preprocessing": r["preprocessing"],
            "resize": _resize_label(r.get("resize")),
            "manual_features": r.get("manual_features", False),
            "train_time_s": r.get("train_time", 0),
        }
        test_results = r.get("test_results", {})

        # New format: test_results = {"best": {...}, "last": {...}}
        if "best" in test_results or "last" in test_results:
            for ckpt_tag in ("best", "last"):
                ckpt_data = test_results.get(ckpt_tag, {})
                for group_name, group_data in ckpt_data.items():
                    if isinstance(group_data, dict):
                        row[f"precision_{ckpt_tag}_{group_name}"] = group_data.get("avg_precision", 0)
        else:
            # Backwards compat with old single-checkpoint format
            for group_name, group_data in test_results.items():
                if isinstance(group_data, dict):
                    row[f"precision_best_{group_name}"] = group_data.get("avg_precision", 0)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison bar charts (using best-checkpoint columns)."""
    best_cols = [c for c in df.columns if c.startswith("precision_best_")]
    if not best_cols:
        return

    # ── Chart 1: all experiments, best checkpoint, by distortion group ──
    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(df))
    width = 0.8 / len(best_cols)

    for i, col in enumerate(best_cols):
        label = col.replace("precision_best_", "")
        ax.bar(x + i * width, df[col].values, width, label=label)

    ax.set_xlabel("Experiment")
    ax.set_ylabel("Average Precision (IoU >= 0.5)")
    ax.set_title("Model Comparison (best checkpoint) - Test Precision by Distortion")
    ax.set_xticks(x + width * len(best_cols) / 2)
    ax.set_xticklabels(df["experiment_id"].values, rotation=45, ha="right", fontsize=6)
    ax.legend(title="Distortion")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(str(output_dir / "model_comparison.png"), dpi=150)
    plt.close(fig)

    # ── Chart 2: best result per model architecture ──
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    mixed_col = "precision_best_mixed"
    if mixed_col in df.columns:
        best_per_model = df.groupby("model")[mixed_col].max().sort_values(ascending=False)
        best_per_model.plot(kind="bar", ax=ax2, color="steelblue")
        ax2.set_ylabel("Best Precision (mixed distortions, best ckpt)")
        ax2.set_title("Best Result per Model Architecture")
        ax2.grid(True, alpha=0.3, axis="y")
        fig2.tight_layout()
        fig2.savefig(str(output_dir / "best_per_model.png"), dpi=150)

    plt.close(fig2)

    # ── Chart 3: best vs last checkpoint comparison ──
    last_mixed = "precision_last_mixed"
    if mixed_col in df.columns and last_mixed in df.columns:
        fig3, ax3 = plt.subplots(figsize=(14, 6))
        x = np.arange(len(df))
        w = 0.35
        ax3.bar(x - w / 2, df[mixed_col].values, w, label="best checkpoint", color="steelblue")
        ax3.bar(x + w / 2, df[last_mixed].values, w, label="last checkpoint", color="coral")
        ax3.set_xlabel("Experiment")
        ax3.set_ylabel("Precision (mixed distortions)")
        ax3.set_title("Best vs Last Checkpoint on Test Set")
        ax3.set_xticks(x)
        ax3.set_xticklabels(df["experiment_id"].values, rotation=45, ha="right", fontsize=6)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")
        fig3.tight_layout()
        fig3.savefig(str(output_dir / "best_vs_last_checkpoint.png"), dpi=150)
        plt.close(fig3)

    # ── Chart 4: manual features impact ──
    if "manual_features" in df.columns and mixed_col in df.columns:
        _plot_manual_features_impact(df, output_dir)


def _plot_manual_features_impact(df: pd.DataFrame, output_dir: Path):
    """Compare precision with vs without manual features for applicable models."""
    mf_df = df[df["manual_features"].isin([True, False])].copy()
    if mf_df.empty:
        return

    match_cols = ["model", "loss", "pretrained", "preprocessing", "resize"]
    mf_true = mf_df[mf_df["manual_features"] == True].copy()
    mf_false = mf_df[mf_df["manual_features"] == False].copy()

    if mf_true.empty or mf_false.empty:
        return

    merged = pd.merge(
        mf_true, mf_false,
        on=match_cols, suffixes=("_mf", "_nomf"),
        how="inner",
    )
    if merged.empty:
        return

    dist_groups = [c.replace("precision_best_", "") for c in mf_true.columns
                   if c.startswith("precision_best_")]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(merged))
    n_groups = len(dist_groups)
    width = 0.8 / max(n_groups * 2, 1)

    for gi, group in enumerate(dist_groups):
        col_mf = f"precision_best_{group}_mf"
        col_nomf = f"precision_best_{group}_nomf"
        if col_mf not in merged.columns or col_nomf not in merged.columns:
            continue
        offset_mf = (gi * 2) * width
        offset_nomf = (gi * 2 + 1) * width
        ax.bar(x + offset_mf, merged[col_mf].values, width,
               label=f"{group} +mf", alpha=0.85)
        ax.bar(x + offset_nomf, merged[col_nomf].values, width,
               label=f"{group} -mf", alpha=0.5)

    # loss/pretrained/preprocessing are part of the merge keys, so they do NOT get suffixes.
    labels = merged.apply(
        lambda r: f"{r['model']}\n{r['loss']}/{r['pretrained']}/{r['preprocessing']}",
        axis=1,
    )
    ax.set_xticks(x + width * n_groups)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Precision (IoU >= 0.5)")
    ax.set_title("Manual Features Impact: With (+mf) vs Without (-mf)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(str(output_dir / "manual_features_impact.png"), dpi=150)
    plt.close(fig)


def _experiment_result_json_valid(exp_output: Path) -> bool:
    """True if experiment_result.json exists, is non-empty, and parses as JSON."""
    path = exp_output / "experiment_result.json"
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _experiment_has_checkpoint(exp_output: Path) -> bool:
    """True if any model checkpoint artifact exists for this experiment."""
    return (
        (exp_output / "best_model.pt").exists()
        or (exp_output / "last_model.pt").exists()
        or (exp_output / "hog_svm.joblib").exists()
        or any(exp_output.glob("*/weights/best.pt"))  # YOLO/DETR
        or any(exp_output.glob("*/weights/last.pt"))
    )


def _experiment_already_done(exp_output: Path) -> bool:
    """Check whether an experiment has already been trained and evaluated."""
    return _experiment_has_checkpoint(exp_output) and _experiment_result_json_valid(exp_output)


def _load_previous_result(exp_output: Path) -> dict:
    """Load the cached result for an already-completed experiment."""
    with open(exp_output / "experiment_result.json", "r") as f:
        return json.load(f)


def _check_split_exists(cfg: dict, split: str) -> bool:
    """Return True if a specific split directory exists and contains obs files."""
    ds_cfg = cfg["dataset"]
    split_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"] / split
    return split_dir.exists() and bool(list(split_dir.glob("*_obs.npy")))


def _check_dataset_exists(cfg: dict) -> bool:
    return all(_check_split_exists(cfg, s) for s in ("train", "val", "test"))


def _setup_error_log(output_dir: Path) -> logging.Logger:
    """Create a file logger for experiment errors."""
    log_path = output_dir / "error_log.txt"
    logger = logging.getLogger("experiment_errors")
    logger.setLevel(logging.ERROR)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "zombie_detection" / "config.yaml"))
    parser.add_argument(
        "--output-dir",
        default="/media/tristan-toye/ESD-USB/results",
        help="Experiment outputs (checkpoints, experiment_result.json). "
        "Default is USB; use --output-dir to override if the drive is not mounted.",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Only run these models (subset)")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain even if checkpoint exists")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate dataset if missing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    error_logger = _setup_error_log(output_dir)

    # ── Dataset check ──
    if not _check_dataset_exists(cfg):
        if args.generate_dataset:
            print("Dataset not found. Generating...")
            from train.generate_dataset import main as gen_main
            gen_main()
        else:
            ds_cfg = cfg["dataset"]
            data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]
            print(f"ERROR: Dataset not found at {data_dir}")
            print("Run with --generate-dataset to auto-generate, or run:")
            print("  python -m train.generate_dataset")
            sys.exit(1)

    experiments = generate_experiment_configs(cfg)

    if args.models:
        experiments = [e for e in experiments if e["model"] in args.models]

    total = len(experiments)
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT PIPELINE: {total} experiments queued")
    print(f"  Output: {output_dir}")
    print(f"  Error log: {output_dir / 'error_log.txt'}")
    print(f"{'='*70}\n")

    all_results = []
    skipped = 0
    succeeded = 0
    failed_experiments = []
    pipeline_start = time.time()

    exp_bar = tqdm(experiments, desc="Experiments", unit="exp", position=0)
    for i, exp in enumerate(exp_bar):
        exp_bar.set_description(f"[{i+1}/{total}] {exp['model']}")

        exp_output = output_dir / exp["id"]
        exp_output.mkdir(parents=True, exist_ok=True)

        # Skip if already trained and evaluated
        if not args.force_retrain and _experiment_already_done(exp_output):
            exp_bar.write(f"  SKIP  {exp['id']}  (cached)")
            result = _load_previous_result(exp_output)
            all_results.append(result)
            skipped += 1
            continue

        # If a checkpoint exists but results are missing/invalid, run TEST only.
        if (
            not args.force_retrain
            and _experiment_has_checkpoint(exp_output)
            and not _experiment_result_json_valid(exp_output)
        ):
            mf = exp.get("manual_features", False)
            exp_bar.write(f"\n  TEST-ONLY {exp['id']}  (checkpoint exists, no valid results)")
            exp_bar.write(f"        model={exp['model']}  loss={exp['loss']}  "
                          f"pretrained={exp['pretrained']}  preproc={exp['preprocessing']}  "
                          f"resize={_resize_label(exp.get('resize'))}  mf={mf}")

            exp_failed = False
            test_results = {}
            try:
                test_results = evaluate_on_test(
                    model_name=exp["model"],
                    checkpoint_dir=exp_output,
                    cfg=cfg,
                    preprocessing_variant=exp["preprocessing"],
                    pretrained_mode=exp["pretrained"],
                    resize=exp.get("resize"),
                    manual_features=mf,
                )
            except Exception as e:
                tb = traceback.format_exc()
                exp_bar.write(f"  TEST FAILED: {e}")
                error_logger.error(
                    f"TEST FAILED (TEST-ONLY) | {exp['id']}\n{tb}\n{'─'*60}"
                )
                test_results = {"error": str(e)}
                exp_failed = True

            result = {
                **exp,
                "status": "failed" if exp_failed else "success",
                "train_time": 0.0,
                "train_results": {"skipped": "checkpoint_exists"},
                "test_results": test_results,
            }
            all_results.append(result)

            if not exp_failed:
                with open(exp_output / "experiment_result.json", "w") as f:
                    json.dump(result, f, indent=2, default=str)
                succeeded += 1
                exp_bar.write(f"  DONE  {exp['id']}  (test-only)")
            else:
                failed_experiments.append(exp["id"])

            with open(output_dir / "experiments_results.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            exp_bar.set_postfix(ok=succeeded, fail=len(failed_experiments), skip=skipped)
            continue

        mf = exp.get("manual_features", False)
        exp_bar.write(f"\n  START {exp['id']}")
        exp_bar.write(f"        model={exp['model']}  loss={exp['loss']}  "
                       f"pretrained={exp['pretrained']}  preproc={exp['preprocessing']}  "
                       f"resize={_resize_label(exp.get('resize'))}  mf={mf}")

        exp_failed = False
        start = time.time()

        # ── Training ──
        try:
            train_results = train_model(
                model_name=exp["model"],
                cfg=cfg,
                output_dir=exp_output,
                loss_name=exp["loss"],
                preprocessing_variant=exp["preprocessing"],
                pretrained_mode=exp["pretrained"],
                resize=exp.get("resize"),
                manual_features=mf,
            )
        except Exception as e:
            tb = traceback.format_exc()
            exp_bar.write(f"  TRAIN FAILED: {e}")
            error_logger.error(
                f"TRAIN FAILED | {exp['id']}\n{tb}\n{'─'*60}"
            )
            train_results = {"error": str(e)}
            exp_failed = True

        train_time = time.time() - start

        # ── Test evaluation ──
        test_results = {}
        if not exp_failed:
            try:
                test_results = evaluate_on_test(
                    model_name=exp["model"],
                    checkpoint_dir=exp_output,
                    cfg=cfg,
                    preprocessing_variant=exp["preprocessing"],
                    pretrained_mode=exp["pretrained"],
                    resize=exp.get("resize"),
                    manual_features=mf,
                )
            except Exception as e:
                tb = traceback.format_exc()
                exp_bar.write(f"  TEST FAILED: {e}")
                error_logger.error(
                    f"TEST FAILED | {exp['id']}\n{tb}\n{'─'*60}"
                )
                test_results = {"error": str(e)}
                exp_failed = True

        result = {
            **exp,
            "status": "failed" if exp_failed else "success",
            "train_time": train_time,
            "train_results": train_results,
            "test_results": test_results,
        }
        all_results.append(result)

        # Save per-experiment result for skip-on-rerun
        if not exp_failed:
            with open(exp_output / "experiment_result.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
            succeeded += 1
            exp_bar.write(f"  DONE  {exp['id']}  ({_fmt_duration(train_time)})")
        else:
            failed_experiments.append(exp["id"])

        # Save incremental global results
        with open(output_dir / "experiments_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        exp_bar.set_postfix(ok=succeeded, fail=len(failed_experiments), skip=skipped)

    # ── Final summary ──
    pipeline_time = time.time() - pipeline_start
    ran = total - skipped
    n_failed = len(failed_experiments)

    df = create_comparison_table(all_results)
    df.to_csv(str(output_dir / "comparison.csv"), index=False)
    plot_comparison(df, output_dir)

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE  ({_fmt_duration(pipeline_time)} total)")
    print(f"{'='*70}")
    print(f"  Total experiments:  {total}")
    print(f"  Skipped (cached):   {skipped}")
    print(f"  Ran this session:   {ran}")
    print(f"  Succeeded:          {succeeded}")
    print(f"  Failed:             {n_failed}")
    print()
    print(f"  Results JSON:  {output_dir / 'experiments_results.json'}")
    print(f"  Comparison:    {output_dir / 'comparison.csv'}")
    print(f"  Charts:        {output_dir / 'model_comparison.png'}")
    if n_failed:
        print(f"  Error log:     {output_dir / 'error_log.txt'}")

    if n_failed:
        print(f"\n  FAILED EXPERIMENTS ({n_failed}):")
        for fid in failed_experiments:
            print(f"    - {fid}")

    print(f"{'='*70}")

    # Print top 5 by mixed precision (best checkpoint)
    best_mixed = "precision_best_mixed"
    last_mixed = "precision_last_mixed"
    if best_mixed in df.columns and not df[best_mixed].isna().all():
        cols = ["model", "loss", "pretrained", "preprocessing", "resize", best_mixed]
        if last_mixed in df.columns:
            cols.append(last_mixed)
        top5 = df.nlargest(5, best_mixed)[cols]
        print("\n  TOP 5 (mixed distortions, best checkpoint):")
        print(top5.to_string(index=False))
        print()

    # ── Feature ablation on manual-features experiments ──
    mf_results = [r for r in all_results
                  if r.get("manual_features") and r.get("status") == "success"]
    if mf_results:
        print(f"\n{'='*70}")
        print("  FEATURE ABLATION (test-time only)")
        print(f"{'='*70}")
        run_feature_ablation(mf_results, cfg, output_dir)


# ──────────────────────── Feature Ablation ────────────────────────


def run_feature_ablation(
    mf_experiments: list[dict],
    cfg: dict,
    output_dir: Path,
):
    """Run leave-one-out and single-feature ablation on trained manual-features models.

    For each experiment, loads the best checkpoint and evaluates 8 times
    with different channel masks on the test set.
    """
    from train.preprocessing import MANUAL_FEATURE_NAMES, NUM_MANUAL_FEATURES

    all_rows = []

    for exp in tqdm(mf_experiments, desc="Ablation experiments", unit="exp"):
        exp_output = output_dir / exp["id"]
        model_name = exp["model"]
        preproc = exp["preprocessing"]
        pretrained = exp["pretrained"]
        resize = exp.get("resize")

        baseline_results = exp.get("test_results", {}).get("best", {})

        # Leave-one-out: drop one feature at a time
        for fi, fname in enumerate(MANUAL_FEATURE_NAMES):
            mask = [True] * NUM_MANUAL_FEATURES
            mask[fi] = False
            ablation_results = _evaluate_checkpoint(
                model_name, exp_output, cfg,
                preproc, pretrained, resize,
                checkpoint_name="best_model.pt",
                manual_features=True,
                manual_features_mask=mask,
            )
            for group_name, gdata in ablation_results.items():
                if not isinstance(gdata, dict):
                    continue
                baseline_prec = baseline_results.get(group_name, {}).get("avg_precision", 0)
                ablation_prec = gdata.get("avg_precision", 0)
                all_rows.append({
                    "experiment_id": exp["id"],
                    "model": model_name,
                    "ablation_type": "leave_one_out",
                    "feature_removed": fname,
                    "distortion_group": group_name,
                    "baseline_precision": baseline_prec,
                    "ablation_precision": ablation_prec,
                    "precision_delta": ablation_prec - baseline_prec,
                })

        # Single-feature: keep only one feature at a time
        for fi, fname in enumerate(MANUAL_FEATURE_NAMES):
            mask = [False] * NUM_MANUAL_FEATURES
            mask[fi] = True
            ablation_results = _evaluate_checkpoint(
                model_name, exp_output, cfg,
                preproc, pretrained, resize,
                checkpoint_name="best_model.pt",
                manual_features=True,
                manual_features_mask=mask,
            )
            for group_name, gdata in ablation_results.items():
                if not isinstance(gdata, dict):
                    continue
                baseline_prec = baseline_results.get(group_name, {}).get("avg_precision", 0)
                ablation_prec = gdata.get("avg_precision", 0)
                all_rows.append({
                    "experiment_id": exp["id"],
                    "model": model_name,
                    "ablation_type": "single_feature",
                    "feature_kept": fname,
                    "distortion_group": group_name,
                    "baseline_precision": baseline_prec,
                    "ablation_precision": ablation_prec,
                    "precision_delta": ablation_prec - baseline_prec,
                })

    if not all_rows:
        return

    abl_df = pd.DataFrame(all_rows)
    abl_df.to_csv(str(output_dir / "feature_ablation.csv"), index=False)

    with open(output_dir / "feature_ablation_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    _plot_ablation(abl_df, output_dir)
    print(f"  Ablation CSV:    {output_dir / 'feature_ablation.csv'}")
    print(f"  Ablation charts: {output_dir / 'ablation_leave_one_out.png'}")
    print(f"                   {output_dir / 'ablation_single_feature.png'}")


def _plot_ablation(abl_df: pd.DataFrame, output_dir: Path):
    """Generate ablation bar charts."""
    # ── Leave-one-out chart ──
    loo = abl_df[abl_df["ablation_type"] == "leave_one_out"].copy()
    if not loo.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = loo.pivot_table(
            index="feature_removed", columns="distortion_group",
            values="precision_delta", aggfunc="mean",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Precision Delta (ablated - baseline)")
        ax.set_xlabel("Feature Removed")
        ax.set_title("Leave-One-Out Ablation: Precision Drop per Feature")
        ax.legend(title="Distortion", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(str(output_dir / "ablation_leave_one_out.png"), dpi=150)
        plt.close(fig)

    # ── Single-feature chart ──
    sf = abl_df[abl_df["ablation_type"] == "single_feature"].copy()
    if not sf.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot = sf.pivot_table(
            index="feature_kept", columns="distortion_group",
            values="ablation_precision", aggfunc="mean",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Precision (single feature active)")
        ax.set_xlabel("Feature Kept")
        ax.set_title("Single-Feature Ablation: Precision with Each Feature Alone")
        ax.legend(title="Distortion", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(str(output_dir / "ablation_single_feature.png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
