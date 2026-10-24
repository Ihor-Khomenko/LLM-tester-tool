"""Individual B/W chart builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from llm_tester.figures.style import (
    apply_thesis_style,
    figure_size,
    gray_for,
    hatch_for,
    labels_for,
    linestyle_for,
    marker_for,
    save_figure,
)

import matplotlib.pyplot as plt


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.size == 0:
        return None
    arr = np.sort(arr)
    if arr.size == 1:
        return float(arr[0])
    k = (arr.size - 1) * (p / 100.0)
    f = int(np.floor(k))
    c = int(np.ceil(k))
    if f == c:
        return float(arr[f])
    return float(arr[f] + (arr[c] - arr[f]) * (k - f))


def plot_asr_bars(
    rows: Sequence[Dict[str, Any]],
    out_base: Path,
    *,
    lang: str = "uk",
) -> Tuple[Path, Path]:
    """Grouped bars: ASR with vs without guard per model/dataset."""
    apply_thesis_style(lang)
    L = labels_for(lang)
    labels = [str(r.get("label") or r.get("model") or r.get("run") or i) for i, r in enumerate(rows)]
    with_g = [float(r.get("asr_with_guard") or r.get("asr") or 0.0) for r in rows]
    without = [
        float(
            r.get("asr_without_guard")
            if r.get("asr_without_guard") is not None
            else (1.0 if r.get("asr_with_guard") is not None else (r.get("asr") or 0.0))
        )
        for r in rows
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=figure_size())
    b1 = ax.bar(
        x - width / 2,
        without,
        width,
        label=L["without_guard"],
        facecolor="white",
        edgecolor="black",
        hatch=hatch_for(0),
    )
    b2 = ax.bar(
        x + width / 2,
        with_g,
        width,
        label=L["with_guard"],
        facecolor="0.85",
        edgecolor="black",
        hatch=hatch_for(1),
    )
    ax.set_ylabel(L["asr"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="best")
    ax.set_xlabel(L["model"] if any(r.get("model") for r in rows) else L["run"])
    fig.tight_layout()
    return save_figure(fig, out_base)


def plot_stage_ablation(
    rows: Sequence[Dict[str, Any]],
    out_base: Path,
    *,
    lang: str = "uk",
) -> Tuple[Path, Path]:
    """Grouped bars for TPR / FPR / F1 across cascade ablations."""
    apply_thesis_style(lang)
    L = labels_for(lang)
    stages = [str(r.get("ablation") or r.get("label") or r.get("run")) for r in rows]
    metrics = [("tpr", L["tpr"]), ("fpr", L["fpr"]), ("f1", L["f1"])]
    x = np.arange(len(stages))
    width = 0.25
    fig, ax = plt.subplots(figsize=figure_size())
    for i, (key, name) in enumerate(metrics):
        vals = [float(r.get(key) or 0.0) for r in rows]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=name,
            facecolor=gray_for(i),
            edgecolor="black",
            hatch=hatch_for(i),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=20, ha="right")
    ax.set_ylabel(L["value"])
    ax.set_xlabel(L["stage"])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, out_base)


def plot_latency_box(
    series: Dict[str, Sequence[float]],
    out_base: Path,
    *,
    lang: str = "uk",
    unit: str = "ms",
) -> Tuple[Path, Path]:
    """Box plot of latency distributions with p50/p95 markers in legend text."""
    apply_thesis_style(lang)
    L = labels_for(lang)
    labels = list(series.keys())
    data = [list(series[k]) for k in labels]
    fig, ax = plt.subplots(figsize=figure_size())
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "white", "edgecolor": "black"},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_hatch(hatch_for(i))
    # Mark p50 / p95 as horizontal lines annotated.
    for i, vals in enumerate(data, start=1):
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        if p50 is not None:
            ax.plot(i, p50, marker=marker_for(0), color="black", markersize=6, linestyle="None")
        if p95 is not None:
            ax.plot(i, p95, marker=marker_for(1), color="black", markersize=6, linestyle="None")
    ylab = L["latency_ms"] if unit == "ms" else L["latency_s"]
    ax.set_ylabel(ylab)
    ax.set_xlabel(L["run"])
    # Custom legend for markers
    ax.plot([], [], marker=marker_for(0), color="black", linestyle="None", label=L["p50"])
    ax.plot([], [], marker=marker_for(1), color="black", linestyle="None", label=L["p95"])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, out_base)


def plot_latency_cdf(
    series: Dict[str, Sequence[float]],
    out_base: Path,
    *,
    lang: str = "uk",
    unit: str = "ms",
) -> Tuple[Path, Path]:
    apply_thesis_style(lang)
    L = labels_for(lang)
    fig, ax = plt.subplots(figsize=figure_size())
    for i, (name, vals) in enumerate(series.items()):
        arr = np.sort(np.asarray(list(vals), dtype=float))
        if arr.size == 0:
            continue
        y = np.arange(1, arr.size + 1) / arr.size
        ax.plot(
            arr,
            y,
            label=name,
            color=gray_for(i),
            linestyle=linestyle_for(i),
            marker=marker_for(i),
            markevery=max(1, arr.size // 12),
            markersize=4,
        )
        p50 = _percentile(arr, 50)
        p95 = _percentile(arr, 95)
        if p50 is not None:
            ax.axvline(p50, color=gray_for(i), linestyle=":", linewidth=0.8)
        if p95 is not None:
            ax.axvline(p95, color=gray_for(i), linestyle="--", linewidth=0.8)
    ax.set_xlabel(L["latency_ms"] if unit == "ms" else L["latency_s"])
    ax.set_ylabel(L["cdf"])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, out_base)


def plot_escalation(
    rows: Sequence[Dict[str, Any]],
    out_base: Path,
    *,
    lang: str = "uk",
) -> Tuple[Path, Path]:
    apply_thesis_style(lang)
    L = labels_for(lang)
    labels = [str(r.get("label") or r.get("ablation") or r.get("run")) for r in rows]
    vals = [float(r.get("escalation_rate") or 0.0) for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figure_size(height_mm=90))
    ax.bar(
        x,
        vals,
        facecolor="white",
        edgecolor="black",
        hatch=hatch_for(2),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(L["escalation"])
    ax.set_ylim(0, max(1.05, max(vals) * 1.15 if vals else 1.05))
    ax.set_xlabel(L["run"])
    fig.tight_layout()
    return save_figure(fig, out_base)


def plot_roc_threshold(
    scores: Sequence[float],
    labels_bin: Sequence[int],
    out_base: Path,
    *,
    lang: str = "uk",
) -> Optional[Tuple[Path, Path]]:
    """ROC curve from risk scores + binary labels (1=attack)."""
    if len(scores) < 2 or len(scores) != len(labels_bin):
        return None
    if len(set(int(x) for x in labels_bin)) < 2:
        return None
    apply_thesis_style(lang)
    L = labels_for(lang)
    sc = np.asarray(scores, dtype=float)
    y = np.asarray(labels_bin, dtype=int)
    thresholds = np.unique(sc)
    if thresholds.size < 2:
        thresholds = np.linspace(float(sc.min()), float(sc.max()), num=20)
    tprs: List[float] = []
    fprs: List[float] = []
    for thr in sorted(thresholds, reverse=True):
        pred = (sc >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tprs.append(tp / (tp + fn) if (tp + fn) else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
    # Add endpoints
    fprs = [0.0] + fprs + [1.0]
    tprs = [0.0] + tprs + [1.0]
    fig, ax = plt.subplots(figsize=figure_size())
    ax.plot(
        fprs,
        tprs,
        color="black",
        linestyle="-",
        marker="o",
        markersize=3,
        markevery=max(1, len(fprs) // 15),
        label="ROC",
    )
    ax.plot([0, 1], [0, 1], color="0.5", linestyle="--", label="chance")
    ax.set_xlabel(L["fpr_axis"])
    ax.set_ylabel(L["tpr_axis"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, out_base)


def plot_run_comparison(
    rows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    out_base: Path,
    *,
    lang: str = "uk",
) -> Tuple[Path, Path]:
    """Grouped bars comparing selected metrics across runs."""
    apply_thesis_style(lang)
    L = labels_for(lang)
    run_labels = [str(r.get("label") or r.get("run")) for r in rows]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=figure_size())
    for i, r in enumerate(rows):
        vals = [float(r.get(m) or 0.0) for m in metrics]
        ax.bar(
            x + i * width,
            vals,
            width,
            label=run_labels[i],
            facecolor=gray_for(i),
            edgecolor="black",
            hatch=hatch_for(i),
        )
    ax.set_xticks(x + width * (len(rows) - 1) / 2)
    ax.set_xticklabels(list(metrics), rotation=20, ha="right")
    ax.set_ylabel(L["value"])
    ax.set_xlabel(L["metric"])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return save_figure(fig, out_base)
