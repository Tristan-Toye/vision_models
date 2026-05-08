#!/usr/bin/env python3
"""Time the zombie detector on **live** KAZ frames (random archer actions, no RL agent).

This steps the PettingZoo environment the same way as ``evaluation.py``, but the
policy is uniform random. Only **RGB (H,W,3) uint8** observations are timed.

Examples::

    cd /path/to/RL-KAZ
    PYTHONPATH=. python -m zombie_detection.realtime.bench_live_env \\
        --steps 400 --submission-config submission_config.yaml \\
        --plot timing_live.png
    # → timing_live.png, timing_live_stats.csv, and a markdown table on stdout

    # Headless, fixed seed
    PYTHONPATH=. python -m zombie_detection.realtime.bench_live_env -n 200 --seed 0

    # With window (slower; includes render cost in wall time, not in detector timing)
    PYTHONPATH=. python -m zombie_detection.realtime.bench_live_env -n 100 --screen
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

# Repo root on PYTHONPATH (same as evaluation.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import create_environment


def _is_rgb_hwc_uint8(obs: object) -> bool:
    return (
        isinstance(obs, np.ndarray)
        and obs.dtype == np.uint8
        and obs.ndim == 3
        and obs.shape[2] == 3
    )


def _timing_markdown_table(
    pipeline_id: str, n_frames: int, mean_ms: float, std_ms: float
) -> str:
    return (
        "| pipeline_id | n_frames | mean_ms | std_ms |\n"
        "|---|---:|---:|---:|\n"
        f"| {pipeline_id} | {n_frames} | {mean_ms:.4f} | {std_ms:.4f} |"
    )


def _write_timing_csv(
    path: Path, pipeline_id: str, n_frames: int, mean_ms: float, std_ms: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline_id", "n_frames", "mean_ms", "std_ms"])
        w.writerow([pipeline_id, n_frames, mean_ms, std_ms])


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark zombie detector on live KAZ observations")
    ap.add_argument("-n", "--steps", type=int, default=300, help="Max detector timings to collect")
    ap.add_argument("--episodes", type=int, default=5, help="Max episodes before stopping")
    ap.add_argument("--seed", type=int, default=42, help="Master seed (first episode reset)")
    ap.add_argument(
        "--submission-config",
        type=Path,
        default=None,
        help="YAML with active_model (default: <cwd>/submission_config.yaml)",
    )
    ap.add_argument("--distortion", type=int, default=0, help="Visual distortion level 0–5")
    ap.add_argument("--screen", action="store_true", help="Human render mode (optional)")
    ap.add_argument(
        "--plot",
        type=Path,
        default=Path("timing_live.png"),
        help="Output PNG path (matplotlib)",
    )
    ap.add_argument(
        "--table",
        type=Path,
        default=None,
        help="CSV with mean/std (default: same stem as --plot, suffix _stats.csv)",
    )
    args = ap.parse_args()

    cfg_path = args.submission_config
    if cfg_path is None:
        cfg_path = Path.cwd() / "submission_config.yaml"
    cfg_path = cfg_path.expanduser().resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path} (use --submission-config)")

    from zombie_detection.realtime.deployment import load_pipeline_from_submission_config

    pipeline = load_pipeline_from_submission_config(cfg_path)

    from submission import CustomWrapper

    render_mode = "human" if args.screen else None
    env = CustomWrapper(
        create_environment(render_mode=render_mode, distortion_level=args.distortion),
    )

    times_ms: list[float] = []
    steps_seen = 0
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        if steps_seen >= args.steps:
            break
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        first = env.possible_agents[0]
        env.action_space(first).seed(int(rng.integers(0, 2**31 - 1)))

        for agent in env.agent_iter():
            if steps_seen >= args.steps:
                break
            obs, reward, termination, truncation, info = env.last()

            if termination or truncation:
                break

            if _is_rgb_hwc_uint8(obs):
                t0 = time.perf_counter()
                pipeline.detect(obs)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                steps_seen += 1

            action = env.action_space(agent).sample()
            env.step(action)

    env.close()

    if not times_ms:
        raise SystemExit("No RGB observations timed — check wrapper returns (H,W,3) uint8.")

    mean_ms = statistics.mean(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    pid = pipeline.pipeline_id
    n_frames = len(times_ms)

    out = args.plot.expanduser().resolve()
    table_path = args.table.expanduser().resolve() if args.table is not None else out.with_name(f"{out.stem}_stats.csv")
    _write_timing_csv(table_path, pid, n_frames, mean_ms, std_ms)

    print(_timing_markdown_table(pid, n_frames, mean_ms, std_ms))
    print(f"\nWrote table: {table_path}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(times_ms, bins=min(40, max(10, len(times_ms) // 5)), color="steelblue", edgecolor="white")
    ax.axvline(mean_ms, color="darkred", linestyle="--", linewidth=2, label=f"mean = {mean_ms:.2f} ms")
    ax.set_xlabel("Detector latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title(f"Live KAZ — {pipeline.pipeline_id} (random actions, n={len(times_ms)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")


if __name__ == "__main__":
    main()
