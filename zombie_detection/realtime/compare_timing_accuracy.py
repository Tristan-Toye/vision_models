#!/usr/bin/env python3
"""Join live timing CSV with mixed-test accuracy into one table.

Typical usage:
  python -m zombie_detection.realtime.compare_timing_accuracy \
    --timing-csv timing_live_all_n100_stats.csv \
    --out-csv timing_accuracy_join.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Row:
    pipeline_id: str
    n_frames: int
    mean_ms: float
    avg_ms: float
    std_ms: float
    mixed_test_precision: float | None


def _read_timing_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = (row.get("pipeline_id") or "").strip()
            if not pid:
                continue
            rows[pid] = {
                "pipeline_id": pid,
                "n_frames": int(float(row["n_frames"])),
                "mean_ms": float(row["mean_ms"]),
                "avg_ms": float(row.get("avg_ms", row["mean_ms"])),
                "std_ms": float(row["std_ms"]),
            }
    return rows


def _accuracy_map() -> dict[str, float]:
    # Source of truth in this repo: deployment classes embed mixed_test_precision.
    from zombie_detection.realtime.deployment import SUBMISSION_MODEL_REGISTRY

    out: dict[str, float] = {}
    for k, cls in SUBMISSION_MODEL_REGISTRY.items():
        v = getattr(cls, "mixed_test_precision", None)
        if v is None:
            continue
        out[str(k)] = float(v)
    return out


def _as_markdown_table(rows: list[Row]) -> str:
    header = (
        "| model | mixed_test_precision | n_frames | mean_ms | avg_ms | std_ms |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in rows:
        acc = "" if r.mixed_test_precision is None else f"{r.mixed_test_precision:.4f}"
        lines.append(
            f"| {r.pipeline_id} | {acc} | {r.n_frames} | {r.mean_ms:.4f} | {r.avg_ms:.4f} | {r.std_ms:.4f} |"
        )
    return header + "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Combine timing CSV with mixed-test accuracy.")
    ap.add_argument(
        "--timing-csv",
        type=Path,
        required=True,
        help="CSV produced by bench_live_env --all-models --table ...",
    )
    ap.add_argument(
        "--sort",
        type=str,
        default="accuracy_desc",
        choices=["accuracy_desc", "mean_ms_asc", "model"],
        help="Row sort order",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    args = ap.parse_args()

    timing_path = args.timing_csv.expanduser().resolve()
    timing = _read_timing_csv(timing_path)
    acc = _accuracy_map()

    joined: list[Row] = []
    for pid, t in timing.items():
        joined.append(
            Row(
                pipeline_id=pid,
                n_frames=int(t["n_frames"]),
                mean_ms=float(t["mean_ms"]),
                avg_ms=float(t["avg_ms"]),
                std_ms=float(t["std_ms"]),
                mixed_test_precision=acc.get(pid),
            )
        )

    if args.sort == "accuracy_desc":
        joined.sort(key=lambda r: (-1.0 if r.mixed_test_precision is None else -r.mixed_test_precision, r.pipeline_id))
    elif args.sort == "mean_ms_asc":
        joined.sort(key=lambda r: (r.mean_ms, r.pipeline_id))
    else:
        joined.sort(key=lambda r: r.pipeline_id)

    print(_as_markdown_table(joined))

    if args.out_csv is not None:
        out_path = args.out_csv.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "mixed_test_precision", "n_frames", "mean_ms", "avg_ms", "std_ms"])
            for r in joined:
                w.writerow(
                    [
                        r.pipeline_id,
                        "" if r.mixed_test_precision is None else f"{r.mixed_test_precision:.10f}",
                        r.n_frames,
                        r.mean_ms,
                        r.avg_ms,
                        r.std_ms,
                    ]
                )


if __name__ == "__main__":
    main()

