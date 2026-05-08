#!/usr/bin/env python3
"""Compute mean IoU (matched pairs) for the top-5 experiments by mixed precision.

This is a **test-only** utility: it reads an existing ``comparison.csv`` produced
by ``python -m zombie_detection.evaluate_models`` and re-evaluates only the top-5
experiments (best checkpoint) to obtain ``avg_iou_matched`` on the mixed test set.

Usage:

  PYTHONPATH=. python -m zombie_detection.compute_mean_iou_top5 \\
      --results-root /path/to/evaluate_models/output \\
      --out top5_mean_iou.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from zombie_detection.evaluate_models import _evaluate_checkpoint, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True, help="Folder containing comparison.csv and experiments.")
    ap.add_argument("--config", type=Path, default=None, help="Optional zombie_detection/config.yaml override.")
    ap.add_argument("--out", type=Path, default=Path("top5_mean_iou.csv"), help="Output CSV path.")
    args = ap.parse_args()

    root = args.results_root.expanduser().resolve()
    comp = root / "comparison.csv"
    if not comp.exists():
        raise SystemExit(f"Missing {comp} (run zombie_detection.evaluate_models first).")

    default_cfg = Path(__file__).resolve().parents[1] / "zombie_detection" / "config.yaml"
    cfg_path = args.config.expanduser().resolve() if args.config is not None else default_cfg
    cfg = load_config(str(cfg_path))

    df = pd.read_csv(comp)
    if "precision_best_mixed" not in df.columns:
        raise SystemExit("comparison.csv missing precision_best_mixed column.")

    top5 = df.nlargest(5, "precision_best_mixed").copy()
    out_rows: list[dict] = []
    any_images = False

    for _, r in top5.iterrows():
        exp_id = str(r["experiment_id"])
        exp_dir = root / exp_id
        if not exp_dir.is_dir():
            print(f"[skip] missing experiment dir: {exp_dir}")
            continue

        resize = None
        resize_label = str(r.get("resize", "native"))
        if resize_label not in ("native", "None", "nan", "---"):
            try:
                w, h = resize_label.split("x")
                resize = [int(w), int(h)]
            except Exception:
                resize = None

        result = _evaluate_checkpoint(
            model_name=str(r["model"]),
            checkpoint_dir=exp_dir,
            cfg=cfg,
            preprocessing_variant=str(r.get("preprocessing", "rgb")),
            pretrained_mode=str(r.get("pretrained", "finetune")),
            resize=resize,
            checkpoint_name="best_model.pt",
            manual_features=bool(r.get("manual_features", False)),
        )

        mixed = result.get("mixed", {})
        n_mixed = int(mixed.get("num_images", 0)) if isinstance(mixed, dict) else 0
        if n_mixed > 0:
            any_images = True
        out_rows.append(
            {
                "experiment_id": exp_id,
                "model": str(r["model"]),
                "precision_best_mixed": float(r["precision_best_mixed"]),
                "mean_iou_best_mixed": float(mixed.get("avg_iou_matched", 0.0)) if isinstance(mixed, dict) else 0.0,
                "num_images_mixed": n_mixed,
            },
        )

    out_df = pd.DataFrame(out_rows).sort_values("precision_best_mixed", ascending=False)
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if not any_images:
        print(
            "\nWARNING: evaluated 0 test images for 'mixed'.\n"
            "This usually means the test dataset directory was not found.\n"
            "Expected (by default config.yaml): zombie_detection/data/kaz_zombie_v1/test\n"
            "If your dataset lives elsewhere (e.g. on USB), create a symlink, e.g.:\n"
            "  ln -s /media/tristan-toye/ESD-USB/kaz_zombie_v1 zombie_detection/data/kaz_zombie_v1\n"
        )


if __name__ == "__main__":
    main()

