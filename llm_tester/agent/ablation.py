"""Stage / aggregation / checkpoint ablation matrix for paper tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_tester.agent.eval import EvalMetrics, evaluate_traces
from llm_tester.agent.traces import TraceStep
from llm_tester.sentinel.guard import build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy, load_policy

# Cumulative stage stacks requested for the paper.
STAGE_STACKS: Tuple[Tuple[str, Dict[str, bool]], ...] = (
    ("S1", {"enable_s1": True, "enable_s2": False, "enable_s3": False, "enable_s4": False}),
    ("S1+S2", {"enable_s1": True, "enable_s2": True, "enable_s3": False, "enable_s4": False}),
    ("S1+S2+S3", {"enable_s1": True, "enable_s2": True, "enable_s3": True, "enable_s4": False}),
    ("full", {"enable_s1": True, "enable_s2": True, "enable_s3": True, "enable_s4": False}),
)

AGGREGATIONS = ("noisy_or", "max", "weighted")

CHECKPOINT_VARIANTS: Tuple[Tuple[str, Dict[str, bool]], ...] = (
    ("T0+T1", {"enable_t0": True, "enable_t1": True}),
    ("no_T0", {"enable_t0": False, "enable_t1": True}),
    ("no_T1", {"enable_t0": True, "enable_t1": False}),
    ("no_T0_T1", {"enable_t0": False, "enable_t1": False}),
)


@dataclass
class AblationCell:
    name: str
    stages: str
    aggregation: str
    checkpoints: str
    metrics: Dict[str, Any]
    config: Dict[str, Any]


def _apply_stack(policy: SentinelPolicy, flags: Dict[str, bool]) -> SentinelPolicy:
    data = policy.to_dict()
    data.update(flags)
    return SentinelPolicy.from_dict(data)


def iter_ablation_configs(
    base: Optional[SentinelPolicy] = None,
    *,
    aggregations: Sequence[str] = AGGREGATIONS,
    include_checkpoint_ablation: bool = True,
    stage_stacks: Sequence[Tuple[str, Dict[str, bool]]] = STAGE_STACKS,
) -> List[Dict[str, Any]]:
    """Expand the paper ablation grid into concrete policy overrides."""
    root = base or SentinelPolicy()
    configs: List[Dict[str, Any]] = []
    ckpt_vars = CHECKPOINT_VARIANTS if include_checkpoint_ablation else (("T0+T1", {"enable_t0": True, "enable_t1": True}),)
    for stage_name, stage_flags in stage_stacks:
        for agg in aggregations:
            for ckpt_name, ckpt_flags in ckpt_vars:
                # Full checkpoint × aggregation sweep only on the full stage stack
                # to keep the matrix reviewable; other stacks keep T0+T1.
                if stage_name != "full" and ckpt_name != "T0+T1":
                    continue
                if stage_name != "full" and agg != "noisy_or":
                    # Aggregation compare on full stack; stages use noisy_or.
                    continue
                name = f"{stage_name}|{agg}|{ckpt_name}"
                configs.append(
                    {
                        "name": name,
                        "stages": stage_name,
                        "aggregation": agg,
                        "checkpoints": ckpt_name,
                        "flags": {**stage_flags, **ckpt_flags, "risk_aggregation": agg},
                    }
                )
    # Always include a pure stage ladder with default agg/checkpoints.
    return configs


def run_ablation_matrix(
    traces: Sequence[TraceStep],
    *,
    policy: Optional[SentinelPolicy] = None,
    policy_path=None,
    aggregations: Sequence[str] = AGGREGATIONS,
    include_checkpoint_ablation: bool = True,
    collect_records: bool = False,
) -> Dict[str, Any]:
    """
    Run each ablation cell; return summary + per-cell metrics (and optional records).

    Offline — no LLM. S4 stays off in the default matrix.
    """
    base = policy or load_policy(policy_path)
    cells: List[AblationCell] = []
    all_records: Dict[str, List[Dict[str, Any]]] = {}

    for cfg in iter_ablation_configs(
        base,
        aggregations=aggregations,
        include_checkpoint_ablation=include_checkpoint_ablation,
    ):
        pol = _apply_stack(base, cfg["flags"])
        guard = build_sentinel(policy=pol, log_path=None)
        if collect_records:
            metrics, records = evaluate_traces(traces, guard, collect_records=True)
            all_records[cfg["name"]] = records
        else:
            metrics = evaluate_traces(traces, guard)
            assert isinstance(metrics, EvalMetrics)
        cells.append(
            AblationCell(
                name=cfg["name"],
                stages=cfg["stages"],
                aggregation=cfg["aggregation"],
                checkpoints=cfg["checkpoints"],
                metrics=metrics.to_dict(),
                config=dict(cfg["flags"]),
            )
        )

    return {
        "n_cells": len(cells),
        "n_traces": len(traces),
        "cells": [
            {
                "name": c.name,
                "stages": c.stages,
                "aggregation": c.aggregation,
                "checkpoints": c.checkpoints,
                "config": c.config,
                "metrics": c.metrics,
            }
            for c in cells
        ],
        "records_by_cell": all_records if collect_records else None,
    }


PRIMARY_CELL_NAME = "full|noisy_or|T0+T1"


def pick_primary_cell(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Cell written to run-dir ``metrics.json``.

    Prefer the full-system default (``full|noisy_or|T0+T1``). The complete
    matrix always lives in ``ablation_matrix.json`` / ``ablation_summary.csv``.
    """
    cells = list(result.get("cells") or [])
    if not cells:
        return None
    for c in cells:
        if c.get("name") == PRIMARY_CELL_NAME:
            return c
    for c in cells:
        if (
            c.get("stages") == "full"
            and c.get("aggregation") == "noisy_or"
            and c.get("checkpoints") == "T0+T1"
        ):
            return c
    # Last resort: any full stack, else last cell (not first / S1-only).
    for c in reversed(cells):
        if c.get("stages") == "full":
            return c
    return cells[-1]


def format_ablation_table(result: Dict[str, Any]) -> str:
    primary = pick_primary_cell(result)
    primary_name = primary["name"] if primary else "(none)"
    lines = [
        "=== ablation matrix ===",
        f"cells={result['n_cells']} traces={result['n_traces']}",
        f"metrics.json primary cell: {primary_name} "
        f"(full matrix → ablation_matrix.json)",
        f"{'name':<32} {'TPR':>7} {'FPR':>7} {'F1':>7} {'ASR':>7}",
    ]
    for cell in result["cells"]:
        m = cell["metrics"]

        def pct(x):
            return "n/a" if x is None else f"{100.0 * float(x):5.1f}%"

        mark = "*" if primary and cell["name"] == primary["name"] else " "
        lines.append(
            f"{mark}{cell['name']:<31} "
            f"{pct(m.get('tpr')):>7} "
            f"{pct(m.get('fpr')):>7} "
            f"{pct(m.get('f1')):>7} "
            f"{pct(m.get('asr_with_guard')):>7}"
        )
    return "\n".join(lines)
