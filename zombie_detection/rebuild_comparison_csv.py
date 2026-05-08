#!/usr/bin/env python3
"""Rebuild comparison.csv and experiments_results.json from per-experiment JSON files.

Use this if `comparison.csv` was overwritten/truncated (e.g. by a partial test-only run).

It scans `<results_root>/*/experiment_result.json`, loads them, then re-creates:
  - `<results_root>/experiments_results.json` (list of dicts)
  - `<results_root>/comparison.csv` (flat table)

Usage:
  PYTHONPATH=. python -m zombie_detection.rebuild_comparison_csv \
      --results-root /media/tristan-toye/ESD-USB/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zombie_detection.evaluate_models import create_comparison_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True)
    args = ap.parse_args()

    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    result_files = sorted(root.glob("*/experiment_result.json"))
    if not result_files:
        raise SystemExit(f"No */experiment_result.json files under {root}")

    all_results: list[dict] = []
    for p in result_files:
        try:
            all_results.append(json.loads(p.read_text()))
        except Exception as e:
            print(f"[skip] {p}: {e}")

    # Write global JSON
    out_json = root / "experiments_results.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str) + "\n", encoding="utf-8")

    # Write comparison CSV
    df = create_comparison_table(all_results)
    out_csv = root / "comparison.csv"
    df.to_csv(out_csv, index=False)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}  ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()

