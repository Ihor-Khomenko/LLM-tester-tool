"""Rebuild an S4 grey-zone subset from a prior guard-eval run's results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from llm_tester.agent.traces import TraceStep
from llm_tester.artifacts.run_dir import load_metrics, load_results


def dumps_traces(traces: Sequence[TraceStep]) -> str:
    return "".join(json.dumps(t.to_dict(), ensure_ascii=False) + "\n" for t in traces)


def _record_has_s4_stage(rec: Dict[str, Any]) -> bool:
    """True when nested decisions include an S4 stage score (not just s4:* reasons)."""
    for d in rec.get("decisions") or []:
        if not isinstance(d, dict):
            continue
        for s in d.get("stage_scores") or []:
            stage = s.get("stage") if isinstance(s, dict) else getattr(s, "stage", None)
            if stage == "S4":
                return True
        if d.get("stage") == "S4":
            return True
    return False


def select_grey_zone_ids(
    records: Sequence[Dict[str, Any]],
    *,
    filter_mode: str = "escalated",
) -> List[str]:
    """
    Pick trace ids that hit the S4 grey zone.

    ``filter_mode``:
      * ``escalated`` — ``rec["escalated"]`` (default; S4 was invoked)
      * ``s4-stage`` — nested stage_scores contain S4
      * ``both`` — escalated OR s4-stage
    """
    mode = (filter_mode or "escalated").lower().strip()
    ids: List[str] = []
    seen: Set[str] = set()
    for rec in records:
        tid = str(rec.get("id") or rec.get("trace_id") or "")
        if not tid or tid in seen:
            continue
        escalated = bool(rec.get("escalated"))
        has_s4 = _record_has_s4_stage(rec)
        keep = False
        if mode in ("escalated", "escalate"):
            keep = escalated
        elif mode in ("s4-stage", "s4", "stage"):
            keep = has_s4
        elif mode == "both":
            keep = escalated or has_s4
        else:
            raise ValueError(
                f"Unknown filter_mode {filter_mode!r}; "
                "use escalated | s4-stage | both"
            )
        if keep:
            seen.add(tid)
            ids.append(tid)
    return ids


def rebuild_grey_zone_subset(
    run_dir: Path,
    traces: Sequence[TraceStep],
    *,
    out_path: Path,
    filter_mode: str = "escalated",
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Filter ``traces`` to those escalated in ``run_dir`` results; write JSONL.

    Returns a manifest dict (also written beside the JSONL by default).
    """
    run_dir = Path(run_dir)
    records = load_results(run_dir)
    ids = select_grey_zone_ids(records, filter_mode=filter_mode)
    id_set = set(ids)
    selected = [t for t in traces if str(getattr(t, "id", "")) in id_set]
    # Preserve selection order from results when possible.
    by_id = {str(t.id): t for t in selected}
    ordered = [by_id[i] for i in ids if i in by_id]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dumps_traces(ordered), encoding="utf-8")

    metrics = {}
    try:
        metrics = load_metrics(run_dir) or {}
    except Exception:
        metrics = {}

    manifest: Dict[str, Any] = {
        "source_run": str(run_dir.resolve()),
        "filter": filter_mode,
        "n": len(ordered),
        "n_source_results": len(records),
        "ids": [str(t.id) for t in ordered],
        "out_path": str(out_path.resolve()),
        "source_metrics_slice": {
            k: metrics.get(k)
            for k in (
                "n",
                "n_attack",
                "n_benign",
                "escalations",
                "escalation_rate",
                "tpr",
                "fpr",
                "asr_with_guard",
            )
            if k in metrics
        },
    }
    man_path = (
        Path(manifest_path)
        if manifest_path is not None
        else out_path.with_name(out_path.stem + "_manifest.json")
    )
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(man_path.resolve())
    return manifest
