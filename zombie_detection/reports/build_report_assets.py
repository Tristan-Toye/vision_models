#!/usr/bin/env python3
"""Populate zombie_detection/reports/figures/ for experiment_report.tex.

Reads comparison.csv, experiments_results.json, and feature_ablation.csv from
--results-root, copies representative training curves, and generates summary plots.

Usage:
  python zombie_detection/reports/build_report_assets.py \\
      --results-root /media/tristan-toye/ESD-USB/results

Then from zombie_detection/reports/:
  pdflatex experiment_report.tex
"""

from __future__ import annotations

import argparse
import math
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPORTS_DIR / "figures"
DEFAULT_RESULTS = Path("/media/tristan-toye/ESD-USB/results")


def _best_experiment_id(df: pd.DataFrame, model: str) -> str | None:
    sub = df[df["model"] == model].copy()
    if sub.empty:
        return None
    if "precision_best_mixed" not in sub.columns:
        return None
    sub["precision_best_mixed"] = pd.to_numeric(sub["precision_best_mixed"], errors="coerce")
    sub = sub.dropna(subset=["precision_best_mixed"])
    if sub.empty:
        return None
    return str(sub.loc[sub["precision_best_mixed"].idxmax(), "experiment_id"])


def _exp_dir(results_root: Path, exp_id: str) -> Path:
    return results_root / exp_id


def _copy_learning_curve(results_root: Path, exp_id: str, model: str, dest_name: str) -> bool:
    src = _exp_dir(results_root, exp_id) / f"{model}_curves.png"
    if not src.exists():
        return False
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, FIGURES_DIR / dest_name)
    return True


def _plot_simple_learning_from_history(
    results_root: Path, exp_id: str, model: str, dest_name: str,
) -> bool:
    """Re-plot curves from *_history.json (train loss + val precision only).

    This avoids inheriting older curve PNGs that may include val-loss or other
    styling choices.
    """
    hist_path = _exp_dir(results_root, exp_id) / f"{model}_history.json"
    if not hist_path.exists():
        return False
    try:
        hist = json.loads(hist_path.read_text())
        train_loss = hist.get("train_loss", [])
        val_prec = hist.get("val_precision", [])
        if not train_loss or not val_prec:
            return False
        epochs = np.arange(1, min(len(train_loss), len(val_prec)) + 1)
        train_loss = np.asarray(train_loss[: len(epochs)], dtype=float)
        val_prec = np.asarray(val_prec[: len(epochs)], dtype=float)
    except Exception:
        return False

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / dest_name

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, label="Train Loss", color="C0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{model} - Train Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, val_prec, label="Val Precision", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Precision (IoU >= 0.5)")
    ax2.set_title(f"{model} - Val Precision")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return True


def _find_ultralytics_results(exp_dir: Path, model_name: str) -> Path | None:
    direct = exp_dir / model_name / "results.png"
    if direct.exists():
        return direct
    for p in exp_dir.rglob("results.png"):
        if model_name in p.parts:
            return p
    return None


def _plot_from_results_csv(csv_path: Path, out_path: Path, title: str) -> bool:
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    epoch = df["epoch"] if "epoch" in df.columns else np.arange(len(df))
    has_loss = "train/box_loss" in df.columns
    has_map = "metrics/mAP50(B)" in df.columns
    if not has_loss and not has_map:
        return False
    ncols = 2 if (has_loss and has_map) else 1
    fig, axes = plt.subplots(1, ncols, figsize=(12 if ncols == 2 else 6, 4))
    axes = np.atleast_1d(axes).ravel()
    if has_loss:
        axes[0].plot(epoch, df["train/box_loss"], color="C0")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("train/box_loss")
        axes[0].grid(True, alpha=0.3)
    if has_map:
        axm = axes[1] if has_loss and len(axes) > 1 else axes[0]
        axm.plot(epoch, df["metrics/mAP50(B)"], color="green")
        axm.set_xlabel("epoch")
        axm.set_ylabel("metrics/mAP50(B)")
        axm.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return True


def _copy_or_plot_ultralytics(
    results_root: Path, exp_id: str, model_name: str, dest_png: str, title: str,
) -> bool:
    exp_dir = _exp_dir(results_root, exp_id)
    png = _find_ultralytics_results(exp_dir, model_name)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_DIR / dest_png
    if png and png.exists():
        shutil.copy2(png, dest)
        return True
    csv_path = exp_dir / model_name / "results.csv"
    if _plot_from_results_csv(csv_path, dest, title):
        return True
    for csv in exp_dir.rglob("results.csv"):
        if model_name in csv.parts and _plot_from_results_csv(csv, dest, title):
            return True
    return False


def _classical_val_bar(results: list[dict], model: str, dest_name: str, title: str) -> None:
    val_p = None
    for r in results:
        if r.get("model") != model:
            continue
        tr = r.get("train_results") or {}
        if isinstance(tr, dict) and "val_precision" in tr:
            val_p = float(tr["val_precision"])
            break
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    if val_p is not None:
        ax.bar([model], [val_p], color="steelblue")
        ax.set_ylim(0, max(1.0, val_p * 1.1))
        ax.set_ylabel("Mean val precision (IoU)")
    else:
        ax.text(0.5, 0.5, "No val_precision in experiments_results.json", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(FIGURES_DIR / dest_name), dpi=150)
    plt.close(fig)


def _plot_best_per_model(df: pd.DataFrame, path: Path) -> None:
    df = df.copy()
    df["precision_best_mixed"] = pd.to_numeric(df["precision_best_mixed"], errors="coerce")
    best = df.groupby("model", as_index=False)["precision_best_mixed"].max().sort_values(
        "precision_best_mixed", ascending=False,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(best["model"][::-1], best["precision_best_mixed"][::-1], color="steelblue")
    ax.set_xlabel("Best test precision (mixed distortions)")
    ax.set_title("Best precision_best_mixed per model (any configuration)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_preproc_heatmap(df: pd.DataFrame, path: Path) -> None:
    """Max mixed precision by (model, preprocessing) for resized experiments."""
    df = df.copy()
    df["precision_best_mixed"] = pd.to_numeric(df["precision_best_mixed"], errors="coerce")
    sub = df[df["resize"].astype(str) != "native"].dropna(subset=["precision_best_mixed"])
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index="model", columns="preprocessing", values="precision_best_mixed", aggfunc="max",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("Max precision_best_mixed: model × preprocessing (360×640 runs)")
    fig.colorbar(im, ax=ax, label="precision")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_ablation_loo_feature(abl: pd.DataFrame, path: Path) -> None:
    loo = abl[abl["ablation_type"] == "leave_one_out"].copy()
    if loo.empty:
        return
    g = loo.groupby("feature_removed", as_index=False)["precision_delta"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g["feature_removed"], g["precision_delta"], color="coral")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Δ precision vs baseline")
    ax.set_title("Leave-one-out: mean effect of removing each manual feature")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_ablation_sf_feature(abl: pd.DataFrame, path: Path) -> None:
    sf = abl[abl["ablation_type"] == "single_feature"].copy()
    if sf.empty:
        return
    g = sf.groupby("feature_kept", as_index=False)["ablation_precision"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g["feature_kept"], g["ablation_precision"], color="seagreen")
    ax.set_ylabel("Mean precision (single feature only)")
    ax.set_title("Single-feature ablation: using one channel at a time")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def _plot_ablation_delta_by_distortion_model(abl: pd.DataFrame, path: Path) -> None:
    loo = abl[abl["ablation_type"] == "leave_one_out"].copy()
    if loo.empty:
        return
    pivot = loo.pivot_table(
        index="distortion_group", columns="model", values="precision_delta", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Mean Δ precision (leave-one-out)")
    ax.set_title("Ablation impact by test distortion group and backbone")
    ax.legend(title="model", fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build figures for experiment_report.tex")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Directory with comparison.csv (default: {DEFAULT_RESULTS})",
    )
    args = parser.parse_args()
    root: Path = args.results_root.expanduser().resolve()

    if not root.is_dir():
        raise SystemExit(f"Results root not found: {root}")

    comp_path = root / "comparison.csv"
    if not comp_path.exists():
        raise SystemExit(f"Missing {comp_path}")

    df = pd.read_csv(comp_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    has_mixed = "precision_best_mixed" in df.columns
    if not has_mixed:
        print(
            "[WARN] comparison.csv missing precision_best_mixed; "
            "skipping precision-based summary plots/tables. "
            "Re-generate comparison.csv from a full evaluate_models run to restore them.",
        )

    # --- Summary plots (from CSV) ---
    if has_mixed:
        _plot_best_per_model(df, FIGURES_DIR / "summary_best_mixed_per_model.png")
        _plot_preproc_heatmap(df, FIGURES_DIR / "summary_preproc_heatmap.png")

    abl_path = root / "feature_ablation.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path)
        _plot_ablation_loo_feature(abl, FIGURES_DIR / "ablation_loo_mean_by_feature.png")
        _plot_ablation_sf_feature(abl, FIGURES_DIR / "ablation_sf_mean_precision.png")
        _plot_ablation_delta_by_distortion_model(abl, FIGURES_DIR / "ablation_delta_by_distortion_model.png")
        for name in ("ablation_leave_one_out.png", "ablation_single_feature.png"):
            p = root / name
            if p.exists():
                shutil.copy2(p, FIGURES_DIR / name)

    # --- Learning curves: PyTorch models ---
    pytorch_specs = [
        ("heatmap_cnn", "learn_heatmap_cnn.png"),
        ("resnet18_head", "learn_resnet18_head.png"),
        ("resnet50_head", "learn_resnet50_head.png"),
        ("faster_rcnn", "learn_faster_rcnn.png"),
    ]
    for model, dest in pytorch_specs:
        eid = _best_experiment_id(df, model)
        if eid:
            ok = _plot_simple_learning_from_history(root, eid, model, dest)
            if not ok:
                ok = _copy_learning_curve(root, eid, model, dest)
            print(f"[{'OK' if ok else 'MISS'}] {model} -> {dest} (from {eid})")
        else:
            print(f"[SKIP] no row for model={model}")

    # --- YOLO: pick better of v8 / v11 by mixed precision ---
    yolo_model = "yolov11n"
    if _best_experiment_id(df, "yolov11n") is None:
        yolo_model = "yolov8n"
    y_eid = _best_experiment_id(df, yolo_model)
    if y_eid:
        ok = _copy_or_plot_ultralytics(
            root, y_eid, yolo_model, "learn_yolo.png", f"Ultralytics ({yolo_model})",
        )
        print(f"[{'OK' if ok else 'MISS'}] yolo -> learn_yolo.png ({y_eid})")

    r_eid = _best_experiment_id(df, "rt_detr")
    if r_eid:
        ok = _copy_or_plot_ultralytics(root, r_eid, "rt_detr", "learn_rt_detr.png", "Ultralytics RT-DETR")
        print(f"[{'OK' if ok else 'MISS'}] rt_detr -> learn_rt_detr.png ({r_eid})")

    # --- Classical: bar from JSON ---
    json_path = root / "experiments_results.json"
    if json_path.exists():
        results = json.loads(json_path.read_text())
        _classical_val_bar(results, "hog_svm", "learn_hog_svm.png", "HOG+SVM (val precision)")
        _classical_val_bar(results, "template_match", "learn_template_match.png", "Template match (val precision)")
        print("[OK] classical val bars -> learn_hog_svm.png, learn_template_match.png")

    # --- Export summary table for LaTeX manual paste (optional helper) ---
    summary = []
    for model in sorted(df["model"].unique()):
        eid = _best_experiment_id(df, model)
        sub = df[df["model"] == model]
        sub = sub.copy()
        if "precision_best_mixed" in sub.columns:
            sub["precision_best_mixed"] = pd.to_numeric(sub["precision_best_mixed"], errors="coerce")
            best_m = sub["precision_best_mixed"].max()
        else:
            best_m = float("nan")
        summary.append({"model": model, "best_mixed": best_m, "best_experiment_id": eid})
    pd.DataFrame(summary).to_csv(FIGURES_DIR / "_summary_best_per_model.csv", index=False)
    print(f"Wrote {FIGURES_DIR / '_summary_best_per_model.csv'}")

    if has_mixed:
        _write_summary_tables_tex(df, FIGURES_DIR / "summary_tables.tex")
        print(f"Wrote {FIGURES_DIR / 'summary_tables.tex'}")
    else:
        print("[SKIP] summary_tables.tex (precision columns missing)")

    _write_extra_tables_tex(
        df,
        results_root=root,
        ablation_csv=root / "feature_ablation.csv",
        out_path=FIGURES_DIR / "extra_tables.tex",
    )
    print(f"Wrote {FIGURES_DIR / 'extra_tables.tex'}")
    print(f"Done. Figures in {FIGURES_DIR}")


def _write_summary_tables_tex(df: pd.DataFrame, path: Path) -> None:
    """LaTeX fragment: best per model + top experiments by mixed precision."""
    df = df.copy()
    df["precision_best_mixed"] = pd.to_numeric(df["precision_best_mixed"], errors="coerce")
    best_rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].dropna(subset=["precision_best_mixed"])
        if sub.empty:
            continue
        i = sub["precision_best_mixed"].idxmax()
        r = sub.loc[i]
        best_rows.append(
            (str(r["model"]), float(r["precision_best_mixed"]), str(r["experiment_id"])),
        )
    best_rows.sort(key=lambda x: -x[1])

    lines = [
        "% Auto-generated by build_report_assets.py — do not edit by hand.",
        r"\providecommand{\SummaryBestMixedTable}{%",
        r"\begin{tabular}{@{}lrp{5.6cm}@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Mixed test} & \textbf{Experiment id} \\",
        r"\midrule",
    ]
    for model, pm, eid in best_rows:
        eid_tex = eid.replace("_", r"\_")
        lines.append(
            f"{model} & {pm:.4f} & \\texttt{{\\scriptsize {eid_tex}}} \\\\",
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

    top = df.nlargest(5, "precision_best_mixed")
    lines.append(r"\providecommand{\SummaryTopFiveTable}{%")
    lines.append(r"\begin{tabular}{@{}lllrrrrrr@{}}")
    lines.append(
        r"\toprule \textbf{Model} & \textbf{Loss} & \textbf{Preproc.} & "
        r"\textbf{n} & \textbf{l} & \textbf{h} & \textbf{e} & \textbf{m} \\ \midrule",
    )
    for _, r in top.iterrows():
        def fcol(c):
            v = pd.to_numeric(r.get(c), errors="coerce")
            return f"{float(v):.4f}" if pd.notna(v) else "---"

        lines.append(
            f"{r['model']} & {r['loss']} & {r['preprocessing']} & "
            f"{fcol('precision_best_none')} & {fcol('precision_best_low')} & "
            f"{fcol('precision_best_high')} & {fcol('precision_best_extreme')} & "
            f"{fcol('precision_best_mixed')} \\\\",
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_set(vals: list[str], *, max_items: int = 4) -> str:
    """Compact pretty-print for a small set in a table cell."""
    vals = [v for v in vals if str(v).strip() not in ("", "nan", "None")]
    if not vals:
        return "---"
    uniq = sorted(set(map(str, vals)))
    if len(uniq) <= max_items:
        return ", ".join(uniq)
    head = ", ".join(uniq[:max_items])
    return f"{head}, +{len(uniq) - max_items} more"


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series(["---"] * len(df), index=df.index)


def _write_extra_tables_tex(
    df: pd.DataFrame,
    *,
    results_root: Path,
    ablation_csv: Path,
    out_path: Path,
) -> None:
    """LaTeX fragment: training-config summary + ablation summaries + extra metrics tables."""
    df = df.copy()

    # --- Training config summary (per model family, show explored values) ---
    cfg_cols = {
        "loss": _safe_col(df, "loss"),
        "pretrained": _safe_col(df, "pretrained"),
        "preprocessing": _safe_col(df, "preprocessing"),
        "resize": _safe_col(df, "resize"),
        "manual_features": _safe_col(df, "manual_features"),
    }

    rows = []
    for model in sorted(df["model"].astype(str).unique()):
        sub = df[df["model"].astype(str) == model]
        rows.append(
            {
                "model": model,
                "n_runs": len(sub),
                "loss": _fmt_set(sub.get("loss", pd.Series(dtype=str)).astype(str).tolist()),
                "pretrained": _fmt_set(sub.get("pretrained", pd.Series(dtype=str)).astype(str).tolist()),
                "preprocessing": _fmt_set(sub.get("preprocessing", pd.Series(dtype=str)).astype(str).tolist()),
                "resize": _fmt_set(sub.get("resize", pd.Series(dtype=str)).astype(str).tolist()),
                "manual_features": _fmt_set(sub.get("manual_features", pd.Series(dtype=str)).astype(str).tolist()),
            },
        )

    lines = [
        "% Auto-generated by build_report_assets.py — do not edit by hand.",
        r"\providecommand{\TrainingConfigTable}{%",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{\#runs} & \textbf{Loss} & \textbf{Pretrained} & \textbf{Preproc.} & \textbf{Resize} & \textbf{Manual feat.}\\",
        r"\midrule",
    ]
    for r in rows:
        model = str(r["model"]).replace("_", r"\_")
        lines.append(
            f"{model} & {r['n_runs']} & {r['loss']} & {r['pretrained']} & {r['preprocessing']} & {r['resize']} & {r['manual_features']} \\\\",
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

    # --- Ablation summaries (optional) ---
    if ablation_csv.exists():
        abl = pd.read_csv(ablation_csv)

        # Leave-one-out (delta) summary
        loo = abl[abl["ablation_type"] == "leave_one_out"].copy()
        if not loo.empty and "feature_removed" in loo.columns and "precision_delta" in loo.columns:
            g = (
                loo.groupby("feature_removed", as_index=False)["precision_delta"]
                .mean()
                .sort_values("precision_delta", ascending=True)
            )
            lines.append(r"\providecommand{\AblationLOOTable}{%")
            lines.append(r"\begin{tabular}{@{}lr@{}}")
            lines.append(r"\toprule \textbf{Removed feature} & \textbf{Mean $\Delta$ precision} \\ \midrule")
            for _, rr in g.iterrows():
                feat = str(rr["feature_removed"]).replace("_", r"\_")
                delta = float(rr["precision_delta"])
                lines.append(f"{feat} & {delta:+.4f} \\\\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

        # Single-feature (absolute precision) summary
        sf = abl[abl["ablation_type"] == "single_feature"].copy()
        if not sf.empty and "feature_kept" in sf.columns and "ablation_precision" in sf.columns:
            g = (
                sf.groupby("feature_kept", as_index=False)["ablation_precision"]
                .mean()
                .sort_values("ablation_precision", ascending=False)
            )
            lines.append(r"\providecommand{\AblationSingleFeatureTable}{%")
            lines.append(r"\begin{tabular}{@{}lr@{}}")
            lines.append(r"\toprule \textbf{Single feature} & \textbf{Mean precision} \\ \midrule")
            for _, rr in g.iterrows():
                feat = str(rr["feature_kept"]).replace("_", r"\_")
                prec = float(rr["ablation_precision"])
                lines.append(f"{feat} & {prec:.4f} \\\\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

    # --- Mean IoU table for top-5 (optional, produced by compute_mean_iou_top5.py) ---
    top5_path = results_root / "top5_mean_iou.csv"
    if top5_path.exists():
        try:
            tdf = pd.read_csv(top5_path)
        except Exception:
            tdf = None
        if tdf is not None and {"model", "precision_best_mixed", "mean_iou_best_mixed"}.issubset(set(tdf.columns)):
            # Build a lookup for (model, mixed_precision) -> mean_iou
            iou_lookup: dict[tuple[str, float], float] = {}
            for _, rr in tdf.iterrows():
                try:
                    k = (str(rr["model"]), float(rr["precision_best_mixed"]))
                    iou_lookup[k] = float(rr["mean_iou_best_mixed"])
                except Exception:
                    continue

            # Combined top-5 table: distortion precisions + mean IoU (mixed)
            if "precision_best_mixed" in df.columns:
                ddf = df.copy()
                ddf["precision_best_mixed"] = pd.to_numeric(ddf["precision_best_mixed"], errors="coerce")
                top = ddf.nlargest(5, "precision_best_mixed")

                def fcol(row, c: str) -> str:
                    v = pd.to_numeric(row.get(c), errors="coerce")
                    return f"{float(v):.4f}" if pd.notna(v) else "---"

                lines.append(r"\providecommand{\MergedTopFiveTable}{%")
                lines.append(r"\begin{tabular}{@{}lllrrrrrrr@{}}")
                lines.append(
                    r"\toprule \textbf{Model} & \textbf{Loss} & \textbf{Preproc.} & "
                    r"\textbf{n} & \textbf{l} & \textbf{h} & \textbf{e} & \textbf{m} & "
                    r"\textbf{Mean IoU (m)} \\ \midrule",
                )
                for _, row in top.iterrows():
                    model = str(row["model"]).replace("_", r"\_")
                    loss = str(row.get("loss", "---")).replace("_", r"\_")
                    pre = str(row.get("preprocessing", "---")).replace("_", r"\_")
                    pm = float(pd.to_numeric(row.get("precision_best_mixed"), errors="coerce") or 0.0)
                    miou = iou_lookup.get((str(row["model"]), pm), float("nan"))
                    miou_s = f"{miou:.4f}" if not np.isnan(miou) else "---"
                    lines.append(
                        f"{model} & {loss} & {pre} & "
                        f"{fcol(row,'precision_best_none')} & {fcol(row,'precision_best_low')} & "
                        f"{fcol(row,'precision_best_high')} & {fcol(row,'precision_best_extreme')} & "
                        f"{fcol(row,'precision_best_mixed')} & {miou_s} \\\\",
                    )
                lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

            lines.append(r"\providecommand{\TopFiveMeanIoUTable}{%")
            lines.append(r"\begin{tabular}{@{}lrrr@{}}")
            lines.append(r"\toprule \textbf{Model} & \textbf{Mixed prec.} & \textbf{Mean IoU} & \textbf{\#images} \\ \midrule")
            for _, rr in tdf.iterrows():
                model = str(rr["model"]).replace("_", r"\_")
                p = float(rr["precision_best_mixed"])
                miou = float(rr["mean_iou_best_mixed"])
                nimg = int(rr.get("num_images_mixed", 0))
                lines.append(f"{model} & {p:.4f} & {miou:.4f} & {nimg} \\\\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"}"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
