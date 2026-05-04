"""Benchmark ``ZombieDetectorPipeline`` latency (run on your target hardware).

Example::

    python -m zombie_detection.realtime.benchmark \\
        --pipeline heatmap_cnn \\
        --checkpoint /path/to/exp_dir_or_best_model.pt \\
        --iterations 100 --warmup 10
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from pathlib import Path

from zombie_detection.realtime.deployment import load_pipeline_from_submission_config
from zombie_detection.realtime.pipelines import PIPELINE_REGISTRY, create_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark zombie detector pipeline FPS / ms")
    p.add_argument(
        "--submission-config",
        type=Path,
        default=None,
        help="Use submission_config.yaml + submission_models/ (ignores --pipeline/--checkpoint).",
    )
    p.add_argument("--pipeline", default=None, choices=sorted(PIPELINE_REGISTRY))
    p.add_argument("--checkpoint", default=None, help="Override RLK_ZOMBIE_CHECKPOINT")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    args = p.parse_args()

    if args.submission_config is not None:
        pipe = load_pipeline_from_submission_config(args.submission_config)
    else:
        if not args.pipeline:
            p.error("Either pass --submission-config or --pipeline")
        pipe = create_pipeline(args.pipeline, checkpoint=args.checkpoint)
    frame = np.random.randint(0, 255, (args.height, args.width, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        pipe.detect(frame)

    times_ms: list[float] = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        pipe.detect(frame)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    print(f"Pipeline:      {pipe.pipeline_id}")
    print(f"Describe:      {pipe.describe()}")
    print(f"Iterations:    {args.iterations}  (warmup {args.warmup})")
    print(f"Frame shape:   {args.height}x{args.width}x3 uint8")
    print(f"Latency mean:  {mean_ms:.2f} ms  (std {stdev_ms:.2f} ms)")
    print(f"Approx FPS:    {fps:.1f}")


if __name__ == "__main__":
    main()
