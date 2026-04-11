"""One-command pipeline: dataset generation + training + evaluation.

Generates the dataset (if not already present), then runs every experiment
in the config matrix, skipping any model that has already been trained.

Usage:
    python -m zombie_detection.run_pipeline
    python -m zombie_detection.run_pipeline --models yolov8n heatmap_cnn
    python -m zombie_detection.run_pipeline --force-retrain
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Full zombie detection pipeline: generate data -> train -> evaluate"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "zombie_detection" / "config.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "zombie_detection" / "experiments"),
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Only run these models (subset of config)",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Retrain models even if checkpoints exist",
    )
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="Skip dataset generation even if missing (will error later)",
    )
    args = parser.parse_args()

    from zombie_detection.train import load_config

    cfg = load_config(args.config)

    # ── Step 1: Dataset ──
    if not args.skip_dataset:
        from zombie_detection.evaluate_models import _check_dataset_exists

        if _check_dataset_exists(cfg):
            ds_cfg = cfg["dataset"]
            data_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]
            print(f"Dataset already exists at {data_dir} -- skipping generation.\n")
        else:
            print("=" * 60)
            print("STEP 1: Generating dataset")
            print("=" * 60)
            from zombie_detection.generate_dataset import generate_split
            import os

            os.environ["SDL_VIDEODRIVER"] = "dummy"

            ds_cfg = cfg["dataset"]
            base_dir = PROJECT_ROOT / ds_cfg["base_dir"] / ds_cfg["name"]
            splits = [
                ("train", ds_cfg["num_train_examples"]),
                ("val", ds_cfg["num_val_examples"]),
                ("test", ds_cfg["num_test_examples"]),
            ]
            for split_name, num_examples in splits:
                generate_split(
                    split_name=split_name,
                    num_examples=num_examples,
                    output_dir=base_dir / split_name,
                    cfg=cfg,
                )
            print("\nDataset generation complete!\n")

    # ── Step 2: Experiments ──
    print("=" * 60)
    print("STEP 2: Running experiments (train + evaluate)")
    print("=" * 60)

    # Re-use evaluate_models.main logic but with already-parsed args
    eval_argv = ["--config", args.config, "--output-dir", args.output_dir, "--generate-dataset"]
    if args.models:
        eval_argv += ["--models"] + args.models
    if args.force_retrain:
        eval_argv.append("--force-retrain")

    sys.argv = ["evaluate_models"] + eval_argv
    from zombie_detection.evaluate_models import main as eval_main
    eval_main()


if __name__ == "__main__":
    main()
