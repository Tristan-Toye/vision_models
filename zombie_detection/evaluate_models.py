"""Experiment pipeline: train and evaluate all model configurations.

Iterates over the experiment matrix defined in config.yaml, trains each
model variant, evaluates on the test set per distortion level, and
produces comparison tables and charts.

Usage:
    python -m zombie_detection.evaluate_models [--config zombie_detection/config.yaml]
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

from zombie_detection.train import train_model, compute_precision_iou, load_config
from zombie_detection.models import SELF_TRAINING_MODELS, HEATMAP_MODELS, get_model, get_target_mode
from zombie_detection.models.heatmap_cnn import heatmap_to_boxes
from zombie_detection.preprocessing import build_preprocessing_pipeline
from zombie_detection.dataset import ZombieDetectionDataset


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

        for loss, pretrained, preproc, resize in product(
            applicable_losses, applicable_pretrained, preproc_variants, model_resize_variants,
        ):
            resize_lbl = _resize_label(resize)
            exp_id = f"{model_name}__{loss}__{pretrained}__{preproc}__{resize_lbl}"
            experiments.append({
                "id": exp_id,
                "model": model_name,
                "loss": loss,
                "pretrained": pretrained,
                "preprocessing": preproc,
                "resize": resize,
            })

    return experiments


def evaluate_on_test(
    model_name: str,
    checkpoint_dir: Path,
    cfg: dict,
    preprocessing_variant: str = "rgb",
    pretrained_mode: str = "finetune",
    resize: "list[int] | None" = None,
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

    distortion_results = {}
    for group in cfg["experiments"]["distortion_eval_groups"]:
        group_name = group["name"]
        group_levels = group["levels"]

        precisions = []
        obs_files = sorted(test_dir.glob("*_obs.npy"))

        for obs_path in tqdm(
            obs_files,
            desc=f"    Test [{checkpoint_name.replace('_model.pt','')}] {group_name}",
            leave=False, unit="img",
        ):
            stem = obs_path.name.replace("_obs.npy", "")
            parts = stem.split("_d")
            if len(parts) < 2:
                continue
            try:
                file_level = int(parts[-1])
            except ValueError:
                continue

            if file_level not in group_levels:
                continue

            box_path = test_dir / f"{stem}_zombies.npy"
            if not box_path.exists():
                continue

            image = np.load(str(obs_path))
            gt_boxes = np.load(str(box_path))

            if model_name in SELF_TRAINING_MODELS:
                detector = get_model(model_name, checkpoint=str(ckpt_path))
                pred_boxes = detector.predict(image)
            elif model_name in HEATMAP_MODELS:
                pred_boxes = _predict_heatmap_model(
                    model_name, checkpoint_dir, image, cfg,
                    preprocessing_variant, pretrained_mode, resize,
                    checkpoint_name=checkpoint_name,
                )
            elif model_name == "faster_rcnn":
                pred_boxes = _predict_fasterrcnn(
                    checkpoint_dir, image, cfg, pretrained_mode, resize,
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
):
    import torch

    in_channels = 4 if preproc_variant == "rgb_edge" else 3
    model_kwargs = {"in_channels": in_channels}
    if model_name in {"resnet18_head", "resnet50_head"}:
        model_kwargs["pretrained"] = pretrained_mode != "scratch"
        model_kwargs["freeze_backbone"] = False

    model = get_model(model_name, **model_kwargs)
    ckpt = checkpoint_dir / checkpoint_name
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    transform = build_preprocessing_pipeline(cfg, variant=preproc_variant, augment=False, resize=resize)
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
    checkpoint_name="best_model.pt",
):
    import torch

    transform = build_preprocessing_pipeline(cfg, variant="rgb", augment=False, resize=resize)
    boxes_dummy = np.zeros((0, 4), dtype=np.float32)
    image_proc, _ = transform(image, boxes_dummy)

    model = get_model("faster_rcnn", pretrained=pretrained_mode != "scratch", freeze_backbone=False)
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


def _experiment_already_done(exp_output: Path) -> bool:
    """Check whether an experiment has already been trained and evaluated."""
    has_checkpoint = (
        (exp_output / "best_model.pt").exists()
        or (exp_output / "hog_svm.joblib").exists()
        or any(exp_output.glob("*/weights/best.pt"))  # YOLO/DETR save here
    )
    has_results = (exp_output / "experiment_result.json").exists()
    return has_checkpoint and has_results


def _load_previous_result(exp_output: Path) -> dict:
    """Load the cached result for an already-completed experiment."""
    with open(exp_output / "experiment_result.json", "r") as f:
        return json.load(f)


def _check_dataset_exists(cfg: dict) -> bool:
    ds_cfg = cfg["dataset"]
    data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists() or not list(split_dir.glob("*_obs.npy")):
            return False
    return True


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
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "zombie_detection" / "experiments"))
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
            from zombie_detection.generate_dataset import main as gen_main
            gen_main()
        else:
            ds_cfg = cfg["dataset"]
            data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]
            print(f"ERROR: Dataset not found at {data_dir}")
            print("Run with --generate-dataset to auto-generate, or run:")
            print("  python -m zombie_detection.generate_dataset")
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

        exp_bar.write(f"\n  START {exp['id']}")
        exp_bar.write(f"        model={exp['model']}  loss={exp['loss']}  "
                       f"pretrained={exp['pretrained']}  preproc={exp['preprocessing']}  "
                       f"resize={_resize_label(exp.get('resize'))}")

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


if __name__ == "__main__":
    main()
