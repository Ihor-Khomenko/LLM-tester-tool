"""Compare metrics.json across two or more run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from llm_tester.artifacts.run_dir import load_metrics

# Prefer these keys in a stable order for thesis tables.
COMPARE_KEYS = [
    "n",
    "n_attack",
    "n_benign",
    "total",
    "evaluated",
    "tp",
    "fp",
    "tn",
    "fn",
    "tpr",
    "fpr",
    "precision",
    "recall",
    "f1",
    "asr",
    "attack_success_rate",
    "asr_with_guard",
    "asr_without_guard",
    "asr_reduction",
    "refusal_rate",
    "escalation_rate",
    "escalations",
    "blocked",
    "refusals",
    "attack_successes",
    "errors",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "mean_latency_s",
]


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _collect_keys(metrics_list: Sequence[Dict[str, Any]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for preferred in COMPARE_KEYS:
        if any(preferred in m and _is_scalar(m.get(preferred)) for m in metrics_list):
            ordered.append(preferred)
            seen.add(preferred)
    for m in metrics_list:
        for k, v in m.items():
            if k in seen or not _is_scalar(v):
                continue
            if k in ("command", "dataset", "provider", "model", "files"):
                continue
            ordered.append(k)
            seen.add(k)
    return ordered


def compare_runs(
    run_dirs: Sequence[Union[str, Path]],
) -> Dict[str, Any]:
    """Return a structured metrics diff for ``run_dirs``."""
    labels: List[str] = []
    metrics_list: List[Dict[str, Any]] = []
    paths: List[str] = []
    for rd in run_dirs:
        p = Path(rd)
        labels.append(p.name)
        paths.append(str(p.resolve()))
        metrics_list.append(load_metrics(p))

    keys = _collect_keys(metrics_list)
    rows: List[Dict[str, Any]] = []
    for key in keys:
        row: Dict[str, Any] = {"metric": key}
        values = []
        for i, m in enumerate(metrics_list):
            val = m.get(key)
            row[labels[i]] = val
            values.append(val)
        if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values[:2]):
            row["delta_b_minus_a"] = float(values[1]) - float(values[0])
        else:
            row["delta_b_minus_a"] = None
        rows.append(row)

    return {
        "runs": labels,
        "paths": paths,
        "metrics": keys,
        "rows": rows,
    }


def _fmt_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) <= 1.0 and v != 0.0:
            return f"{v:.4f}"
        return f"{v:.4g}"
    return str(v)


def format_compare(
    comparison: Dict[str, Any],
    *,
    fmt: str = "md",
    caption: Optional[str] = None,
    table_number: int = 1,
) -> str:
    """Format comparison as md | csv | docx-ready markdown."""
    fmt = (fmt or "md").lower().strip()
    if fmt == "docx-ready":
        fmt = "docx"
    labels: List[str] = list(comparison["runs"])
    rows: List[Dict[str, Any]] = comparison["rows"]

    if fmt == "csv":
        header = ["metric", *labels, "delta_b_minus_a"]
        lines = [",".join(header)]
        for row in rows:
            cells = [_fmt_val(row.get("metric"))]
            for lab in labels:
                cells.append(_fmt_val(row.get(lab)))
            cells.append(_fmt_val(row.get("delta_b_minus_a")))
            lines.append(",".join(cells))
        return "\n".join(lines) + "\n"

    # Markdown / docx-ready
    cap = caption or (
        f"Таблиця 4.{table_number} – Порівняння метрик експериментів "
        f"({', '.join(labels)})"
    )
    header = ["Метрика" if fmt == "docx" else "metric", *labels]
    if len(labels) >= 2:
        header.append("Δ (B−A)" if fmt == "docx" else "delta_b_minus_a")

    sep = "| " + " | ".join("---" for _ in header) + " |"
    head = "| " + " | ".join(header) + " |"
    body_lines = []
    for row in rows:
        cells = [_fmt_val(row.get("metric"))]
        for lab in labels:
            cells.append(_fmt_val(row.get(lab)))
        if len(labels) >= 2:
            cells.append(_fmt_val(row.get("delta_b_minus_a")))
        body_lines.append("| " + " | ".join(cells) + " |")

    if fmt == "docx":
        # Caption above table (Ukrainian thesis convention).
        return "\n".join([cap, "", head, sep, *body_lines, ""])
    return "\n".join([head, sep, *body_lines, "", f"*{cap}*", ""])
