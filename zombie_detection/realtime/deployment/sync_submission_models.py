#!/usr/bin/env python3
"""Copy best experiment artifacts into ``submission_models/`` at repo root.

Usage::

    python zombie_detection/realtime/deployment/sync_submission_models.py \\
        --results-root /path/to/evaluate_models/output
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# experiment_dir_name -> submission_models subfolder
SYNC_MAP = {
    "rt_detr__default__finetune__rgb__native__nomf": "rt_detr",
    "yolov11n__default__finetune__rgb__native__nomf": "yolov11n",
    "yolov8n__default__finetune__rgb__native__nomf": "yolov8n",
    "heatmap_cnn__mse__scratch__rgb__360x640__mf": "heatmap_cnn",
    "resnet18_head__mse__finetune__grayscale__360x640__nomf": "resnet18_head",
    "template_match__default__scratch__rgb__native__nomf": "template_match",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Directory containing experiment subfolders (comparison.csv sibling).",
    )
    args = ap.parse_args()
    root = args.results_root.expanduser().resolve()
    dest_root = Path(__file__).resolve().parents[3] / "submission_models"
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    for exp_id, sub in SYNC_MAP.items():
        src = root / exp_id
        if not src.is_dir():
            print(f"[skip] missing source: {src}")
            continue
        dst = dest_root / sub
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[ok] {exp_id} -> {dst}")


if __name__ == "__main__":
    main()
