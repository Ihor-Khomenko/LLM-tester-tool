"""Aggregate scalar metrics across repeated evaluate_traces runs."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

# Scalars reported as mean ± std under --repeats.
REPEAT_METRIC_KEYS = (
    "tpr",
    "fpr",
    "f1",
    "precision",
    "recall",
    "asr_with_guard",
    "asr_block_only",
    "asr_reduction",
    "escalation_rate",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "llm_calls_per_case",
    "auroc",
    "error_rate",
)


def _mean_std(values: Sequence[float]) -> Dict[str, Optional[float]]:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    n = len(xs)
    mean = statistics.mean(xs)
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0}
    return {"n": n, "mean": mean, "std": statistics.stdev(xs)}


def aggregate_repeat_metrics(
    per_repeat: Sequence[Dict[str, Any]],
    *,
    keys: Sequence[str] = REPEAT_METRIC_KEYS,
) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key in keys:
        vals: List[float] = []
        for m in per_repeat:
            v = m.get(key)
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and math.isnan(v)
            ):
                vals.append(float(v))
        out[key] = _mean_std(vals)
    return out


def format_repeat_aggregate(
    aggregate: Dict[str, Dict[str, Optional[float]]],
    *,
    repeats: int,
    title: str = "guard-eval repeats",
) -> str:
    lines = [f"=== {title} (K={repeats}) ==="]
    for key, stats in aggregate.items():
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None:
            continue
        std_s = "n/a" if std is None else f"{std:.4f}"
        lines.append(f"{key}: mean={mean:.4f}  std={std_s}  n={stats.get('n')}")
    return "\n".join(lines)
