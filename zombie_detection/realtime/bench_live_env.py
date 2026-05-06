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
import yaml

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


def _multi_timing_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "| pipeline_id | n_frames | mean_ms | avg_ms | std_ms |\n|---|---:|---:|---:|---:|\n"
    header = (
        "| pipeline_id | n_frames | mean_ms | avg_ms | std_ms |\n"
        "|---|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['pipeline_id']} | {r['n_frames']} | {r['mean_ms']:.4f} | {r['avg_ms']:.4f} | {r['std_ms']:.4f} |"
        )
    return header + "\n".join(lines)


def _write_timing_csv(
    path: Path, pipeline_id: str, n_frames: int, mean_ms: float, std_ms: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline_id", "n_frames", "mean_ms", "std_ms"])
        w.writerow([pipeline_id, n_frames, mean_ms, std_ms])


def _write_multi_timing_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pipeline_id", "n_frames", "mean_ms", "avg_ms", "std_ms"])
        for r in rows:
            w.writerow([r["pipeline_id"], r["n_frames"], r["mean_ms"], r["avg_ms"], r["std_ms"]])


def _collect_timings_live_env(
    *,
    pipeline,
    steps: int,
    episodes: int,
    seed: int,
    distortion: int,
    screen: bool,
    progress_prefix: str = "",
) -> list[float]:
    """Return a list of detector latencies (ms) on live env observations."""
    from submission import CustomWrapper

    render_mode = "human" if screen else None
    env = CustomWrapper(
        create_environment(render_mode=render_mode, distortion_level=distortion),
    )

    times_ms: list[float] = []
    steps_seen = 0
    rng = np.random.default_rng(seed)
    last_print = 0
    print_every = max(1, min(50, steps // 10 if steps > 0 else 10))

    try:
        for _ep in range(episodes):
            if steps_seen >= steps:
                break
            if progress_prefix:
                print(f"{progress_prefix}episode {_ep + 1}/{episodes} (timed {steps_seen}/{steps})", flush=True)
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            first = env.possible_agents[0]
            env.action_space(first).seed(int(rng.integers(0, 2**31 - 1)))

            for agent in env.agent_iter():
                if steps_seen >= steps:
                    break
                obs, _reward, termination, truncation, _info = env.last()

                if termination or truncation:
                    break

                if _is_rgb_hwc_uint8(obs):
                    t0 = time.perf_counter()
                    pipeline.detect(obs)
                    times_ms.append((time.perf_counter() - t0) * 1000.0)
                    steps_seen += 1
                    if progress_prefix and (steps_seen - last_print) >= print_every:
                        last_print = steps_seen
                        print(f"{progress_prefix}timed {steps_seen}/{steps} frames", flush=True)

                action = env.action_space(agent).sample()
                env.step(action)
    finally:
        env.close()

    return times_ms


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
        "--all-models",
        action="store_true",
        help="Benchmark all submission pipelines (ignores active_model in YAML)",
    )
    ap.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated subset of models when using --all-models (default: all)",
    )
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

    from zombie_detection.realtime.deployment import SUBMISSION_MODEL_REGISTRY, load_pipeline_from_submission_config

    out = args.plot.expanduser().resolve()

    def _plot_histogram(times_ms: list[float], *, title_pid: str, mean_ms: float, out_path: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(times_ms, bins=min(40, max(10, len(times_ms) // 5)), color="steelblue", edgecolor="white")
        ax.axvline(mean_ms, color="darkred", linestyle="--", linewidth=2, label=f"mean = {mean_ms:.2f} ms")
        ax.set_xlabel("Detector latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"Live KAZ — {title_pid} (random actions, n={len(times_ms)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    if not args.all_models:
        pipeline = load_pipeline_from_submission_config(cfg_path)
        print(f"Benchmarking: {pipeline.pipeline_id} (n={args.steps})", flush=True)
        t_start = time.perf_counter()
        times_ms = _collect_timings_live_env(
            pipeline=pipeline,
            steps=args.steps,
            episodes=args.episodes,
            seed=args.seed,
            distortion=args.distortion,
            screen=args.screen,
            progress_prefix=f"[{pipeline.pipeline_id}] ",
        )
        if not times_ms:
            raise SystemExit("No RGB observations timed — check wrapper returns (H,W,3) uint8.")

        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        pid = pipeline.pipeline_id
        n_frames = len(times_ms)

        table_path = args.table.expanduser().resolve() if args.table is not None else out.with_name(f"{out.stem}_stats.csv")
        _write_timing_csv(table_path, pid, n_frames, mean_ms, std_ms)

        print(_timing_markdown_table(pid, n_frames, mean_ms, std_ms))
        print(f"\nWrote table: {table_path}")

        _plot_histogram(times_ms, title_pid=pid, mean_ms=mean_ms, out_path=out)
        print(f"Saved plot: {out}")
        print(f"Done: {pid} (wall {time.perf_counter() - t_start:.2f}s)", flush=True)
        return

    # Multi-model benchmarking
    if args.models.strip():
        wanted = [s.strip().lower() for s in args.models.split(",") if s.strip()]
    else:
        wanted = sorted(SUBMISSION_MODEL_REGISTRY.keys())

    rows: list[dict] = []
    for i, key in enumerate(wanted):
        if key not in SUBMISSION_MODEL_REGISTRY:
            print(f"Skipping unknown model key: {key!r}")
            continue
        # Instantiate pipeline using the same YAML file for device/conf/config_path, but override active_model.
        data = yaml.safe_load(cfg_path.read_text()) or {}
        device_raw = data.get("device", "auto")
        device = str(device_raw).strip() if device_raw is not None else "auto"
        conf = float(data.get("conf_threshold", 0.35))
        cpp = data.get("config_path")
        config_path_kw: str | None = cpp if isinstance(cpp, str) and cpp.strip() else None
        cls = SUBMISSION_MODEL_REGISTRY[key]
        print(f"\n=== [{i + 1}/{len(wanted)}] Benchmarking model: {key} (n={args.steps}) ===", flush=True)
        t_start = time.perf_counter()
        try:
            pipeline = cls(device=device, conf_threshold=conf, config_path=config_path_kw)
        except Exception as e:
            print(f"Skipping {key!r}: failed to initialize pipeline ({type(e).__name__}: {e})")
            continue
        print(f"Initialized: {pipeline.pipeline_id} device={device} conf_threshold={conf}", flush=True)

        times_ms = _collect_timings_live_env(
            pipeline=pipeline,
            steps=args.steps,
            episodes=args.episodes,
            seed=args.seed + i,
            distortion=args.distortion,
            screen=args.screen,
            progress_prefix=f"[{pipeline.pipeline_id}] ",
        )
        if not times_ms:
            print(f"Skipping {key!r}: no RGB observations timed")
            continue

        mean_ms = statistics.mean(times_ms)
        std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
        pid = pipeline.pipeline_id
        n_frames = len(times_ms)
        rows.append(
            {
                "pipeline_id": pid,
                "n_frames": n_frames,
                "mean_ms": float(mean_ms),
                "avg_ms": float(mean_ms),  # average == mean (included per request)
                "std_ms": float(std_ms),
            }
        )
        print(
            f"Done: {pid} n_frames={n_frames} mean_ms={mean_ms:.2f} std_ms={std_ms:.2f} (wall {time.perf_counter() - t_start:.2f}s)",
            flush=True,
        )

    # Stable order: same as registry ordering unless user provided explicit list.
    rows_sorted = rows if args.models.strip() else sorted(rows, key=lambda r: r["pipeline_id"])

    table_path = args.table.expanduser().resolve() if args.table is not None else out.with_name(f"{out.stem}_stats.csv")
    _write_multi_timing_csv(table_path, rows_sorted)
    print(_multi_timing_markdown_table(rows_sorted))
    print(f"\nWrote table: {table_path}")


if __name__ == "__main__":
    main()
