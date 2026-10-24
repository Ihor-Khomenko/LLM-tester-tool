"""Black-and-white matplotlib style for Ukrainian PhD thesis (A4 margins)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# A4 width 210 mm; left 30 + right 15 = 45 → content 165 mm.
# Thesis guidance: max figure width ≈ 158 mm.
MAX_WIDTH_MM = 158.0
DEFAULT_DPI = 300

# Greyscale-safe hatches / markers / linestyles (no colour reliance).
HATCHES: Sequence[str] = ("///", "\\\\\\", "xxx", "...", "+++", "---", "ooo", "***")
MARKERS: Sequence[str] = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES: Sequence[str] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)))
GRAYS: Sequence[str] = ("0.15", "0.35", "0.55", "0.75", "0.0", "0.45", "0.65", "0.25")

LABELS = {
    "uk": {
        "asr": "ASR",
        "asr_with": "ASR з захистом",
        "asr_without": "ASR без захисту",
        "model": "Модель",
        "dataset": "Датасет",
        "stage": "Конфігурація каскаду",
        "tpr": "TPR",
        "fpr": "FPR",
        "f1": "F1",
        "metric": "Метрика",
        "value": "Значення",
        "latency_s": "Затримка, с",
        "latency_ms": "Затримка, мс",
        "cdf": "Емпірична CDF",
        "escalation": "Частка ескалацій",
        "run": "Прогін",
        "threshold": "Поріг",
        "score": "Оцінка ризику",
        "tpr_axis": "True Positive Rate",
        "fpr_axis": "False Positive Rate",
        "with_guard": "з захистом",
        "without_guard": "без захисту",
        "p50": "p50",
        "p95": "p95",
        "count": "Кількість",
    },
    "en": {
        "asr": "ASR",
        "asr_with": "ASR with guard",
        "asr_without": "ASR without guard",
        "model": "Model",
        "dataset": "Dataset",
        "stage": "Cascade configuration",
        "tpr": "TPR",
        "fpr": "FPR",
        "f1": "F1",
        "metric": "Metric",
        "value": "Value",
        "latency_s": "Latency, s",
        "latency_ms": "Latency, ms",
        "cdf": "Empirical CDF",
        "escalation": "Escalation rate",
        "run": "Run",
        "threshold": "Threshold",
        "score": "Risk score",
        "tpr_axis": "True Positive Rate",
        "fpr_axis": "False Positive Rate",
        "with_guard": "with guard",
        "without_guard": "without guard",
        "p50": "p50",
        "p95": "p95",
        "count": "Count",
    },
}


def labels_for(lang: str = "uk") -> dict:
    return LABELS["en"] if (lang or "uk").lower().startswith("en") else LABELS["uk"]


def mm_to_inches(mm: float) -> float:
    return mm / 25.4


def figure_size(width_mm: float = MAX_WIDTH_MM, height_mm: Optional[float] = None) -> Tuple[float, float]:
    w = mm_to_inches(min(width_mm, MAX_WIDTH_MM))
    h = mm_to_inches(height_mm) if height_mm is not None else w * 0.62
    return w, h


def _pick_serif_font() -> str:
    """Prefer Times New Roman; fall back to Liberation / DejaVu Serif."""
    preferred = (
        "Times New Roman",
        "TimesNewRoman",
        "Liberation Serif",
        "DejaVu Serif",
        "Nimbus Roman",
        "FreeSerif",
    )
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    # Last resort: matplotlib default serif family name.
    return "DejaVu Serif"


def apply_thesis_style(lang: str = "uk") -> str:
    """Configure global matplotlib rc for B/W thesis figures. Returns font family used."""
    font = _pick_serif_font()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font, "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.color": "0.7",
            "axes.edgecolor": "0.0",
            "axes.labelcolor": "0.0",
            "xtick.color": "0.0",
            "ytick.color": "0.0",
            "text.color": "0.0",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": DEFAULT_DPI,
            "figure.dpi": 100,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.8,
            "axes.prop_cycle": mpl.cycler(color=list(GRAYS)),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return font


def save_figure(fig: plt.Figure, out_base: Path, *, dpi: int = DEFAULT_DPI) -> Tuple[Path, Path]:
    """Save PNG (300 dpi) and SVG next to ``out_base`` (no extension)."""
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    svg = out_base.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def hatch_for(i: int) -> str:
    return HATCHES[i % len(HATCHES)]


def marker_for(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


def linestyle_for(i: int) -> str:
    return LINESTYLES[i % len(LINESTYLES)]


def gray_for(i: int) -> str:
    return GRAYS[i % len(GRAYS)]
