"""Compare metrics.json across two or more run directories (baseline vs each)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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


def _delta_column_name(baseline: str, other: str) -> str:
    return f"delta_{other}_minus_{baseline}"


def compare_runs(
    run_dirs: Sequence[Union[str, Path]],
) -> Dict[str, Any]:
    """
    Structured metrics diff for ``run_dirs``.

    The **first** run is the baseline. For every other run ``R``, a column
    ``delta_<R>_minus_<baseline>`` holds ``R − baseline`` for numeric metrics.
    With exactly two runs this matches the legacy ``delta_b_minus_a`` alias.
    """
    if not run_dirs:
        raise ValueError("compare_runs requires at least one run directory")

    labels: List[str] = []
    metrics_list: List[Dict[str, Any]] = []
    paths: List[str] = []
    for rd in run_dirs:
        p = Path(rd)
        labels.append(p.name)
        paths.append(str(p.resolve()))
        metrics_list.append(load_metrics(p))

    baseline = labels[0]
    delta_cols = [_delta_column_name(baseline, lab) for lab in labels[1:]]

    keys = _collect_keys(metrics_list)
    rows: List[Dict[str, Any]] = []
    for key in keys:
        row: Dict[str, Any] = {"metric": key}
        values = []
        for i, m in enumerate(metrics_list):
            val = m.get(key)
            row[labels[i]] = val
            values.append(val)

        base_val = values[0]
        for i, lab in enumerate(labels[1:], start=1):
            col = delta_cols[i - 1]
            other = values[i]
            if isinstance(base_val, (int, float)) and isinstance(other, (int, float)):
                row[col] = float(other) - float(base_val)
            else:
                row[col] = None

        # Back-compat alias when comparing exactly two runs.
        if len(labels) == 2:
            row["delta_b_minus_a"] = row.get(delta_cols[0])
        rows.append(row)

    return {
        "runs": labels,
        "paths": paths,
        "baseline": baseline,
        "metrics": keys,
        "delta_columns": delta_cols,
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
    baseline = comparison.get("baseline") or (labels[0] if labels else "")
    delta_cols: List[str] = list(
        comparison.get("delta_columns")
        or [
            _delta_column_name(baseline, lab)
            for lab in labels[1:]
        ]
    )

    if fmt == "csv":
        header = ["metric", *labels, *delta_cols]
        if len(labels) == 2 and "delta_b_minus_a" not in delta_cols:
            header.append("delta_b_minus_a")
        lines = [",".join(header)]
        for row in rows:
            cells = [_fmt_val(row.get("metric"))]
            for lab in labels:
                cells.append(_fmt_val(row.get(lab)))
            for col in delta_cols:
                cells.append(_fmt_val(row.get(col)))
            if len(labels) == 2 and "delta_b_minus_a" not in delta_cols:
                cells.append(_fmt_val(row.get("delta_b_minus_a")))
            lines.append(",".join(cells))
        return "\n".join(lines) + "\n"

    # Markdown / docx-ready
    cap = caption or (
        f"Таблиця 4.{table_number} – Порівняння метрик експериментів "
        f"(baseline={baseline}; {', '.join(labels)})"
    )
    if fmt == "docx":
        header = ["Метрика", *labels]
        for lab in labels[1:]:
            header.append(f"Δ ({lab}−{baseline})")
    else:
        header = ["metric", *labels, *delta_cols]

    sep = "| " + " | ".join("---" for _ in header) + " |"
    head = "| " + " | ".join(header) + " |"
    body_lines = []
    for row in rows:
        cells = [_fmt_val(row.get("metric"))]
        for lab in labels:
            cells.append(_fmt_val(row.get(lab)))
        for col in delta_cols:
            cells.append(_fmt_val(row.get(col)))
        body_lines.append("| " + " | ".join(cells) + " |")

    if fmt == "docx":
        return "\n".join([cap, "", head, sep, *body_lines, ""])
    return "\n".join([head, sep, *body_lines, "", f"*{cap}*", ""])
