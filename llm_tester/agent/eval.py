"""Offline evaluation of MCP-Sentinel on traces (imports sentinel; not part of it)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence  # Any: GuardLike duck typing

from llm_tester.agent.traces import TraceStep
from llm_tester.sentinel.guard import build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.types import Checkpoint, Decision, GuardDecision


def compute_auroc(
    scores: Sequence[float], labels: Sequence[bool]
) -> Optional[float]:
    """
    ROC AUC via Mann–Whitney U (no sklearn dependency).

    ``labels`` True = attack (positive class). Higher ``scores`` → more positive.
    """
    pos = [float(s) for s, y in zip(scores, labels) if y]
    neg = [float(s) for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    # Count pairwise wins + 0.5 ties.
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


def _is_hard_block(decision: Decision) -> bool:
    return decision in (Decision.BLOCK, Decision.ASK_USER)


def _is_intervention(decision: Decision) -> bool:
    """Symmetric: block, ask_user, or sanitize count as an intervention."""
    return decision in (Decision.BLOCK, Decision.ASK_USER, Decision.SANITIZE)


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def tpr(self) -> Optional[float]:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else None

    @property
    def fpr(self) -> Optional[float]:
        denom = self.fp + self.tn
        return (self.fp / denom) if denom else None

    @property
    def precision(self) -> Optional[float]:
        denom = self.tp + self.fp
        return (self.tp / denom) if denom else None

    @property
    def recall(self) -> Optional[float]:
        return self.tpr

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class CheckpointStats:
    n: int = 0
    blocks: int = 0
    sanitizes: int = 0
    allows: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "n": self.n,
            "blocks": self.blocks,
            "sanitizes": self.sanitizes,
            "allows": self.allows,
        }


@dataclass
class EvalMetrics:
    n: int = 0
    n_attack: int = 0
    n_benign: int = 0
    # Primary (back-compat) fields mirror *any-intervention* symmetric scoring.
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    block_only: Confusion = field(default_factory=Confusion)
    any_intervention: Confusion = field(default_factory=Confusion)
    escalations: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    asr_without_guard: float = 1.0
    asr_with_guard: float = 0.0
    by_attack_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    by_checkpoint: Dict[str, CheckpointStats] = field(default_factory=dict)
    first_detection: Dict[str, int] = field(default_factory=dict)
    first_block: Dict[str, int] = field(default_factory=dict)
    # Ranking / cost (paper tables).
    auroc: Optional[float] = None
    llm_calls_total: int = 0
    risk_scores: List[float] = field(default_factory=list)
    attack_labels: List[bool] = field(default_factory=list)

    @property
    def tpr(self) -> Optional[float]:
        return self.any_intervention.tpr

    @property
    def fpr(self) -> Optional[float]:
        return self.any_intervention.fpr

    @property
    def precision(self) -> Optional[float]:
        return self.any_intervention.precision

    @property
    def recall(self) -> Optional[float]:
        return self.any_intervention.recall

    @property
    def f1(self) -> Optional[float]:
        return self.any_intervention.f1

    @property
    def asr_reduction(self) -> Optional[float]:
        if self.asr_without_guard <= 0:
            return None
        return 1.0 - (self.asr_with_guard / self.asr_without_guard)

    @property
    def escalation_rate(self) -> Optional[float]:
        return (self.escalations / self.n) if self.n else None

    @property
    def latency_p50_ms(self) -> Optional[float]:
        return _percentile(self.latencies_ms, 50)

    @property
    def latency_p95_ms(self) -> Optional[float]:
        return _percentile(self.latencies_ms, 95)

    @property
    def latency_mean_ms(self) -> Optional[float]:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else None

    @property
    def llm_calls_per_case(self) -> Optional[float]:
        return (self.llm_calls_total / self.n) if self.n else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "n_attack": self.n_attack,
            "n_benign": self.n_benign,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "tpr": self.tpr,
            "fpr": self.fpr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auroc": self.auroc,
            "block_only": self.block_only.to_dict(),
            "any_intervention": self.any_intervention.to_dict(),
            "asr_without_guard": self.asr_without_guard,
            "asr_with_guard": self.asr_with_guard,
            "asr_reduction": self.asr_reduction,
            "escalation_rate": self.escalation_rate,
            "escalations": self.escalations,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "llm_calls_total": self.llm_calls_total,
            "llm_calls_per_case": self.llm_calls_per_case,
            "by_attack_type": self.by_attack_type,
            "by_checkpoint": {k: v.to_dict() for k, v in self.by_checkpoint.items()},
            "first_detection": dict(self.first_detection),
            "first_block": dict(self.first_block),
            "detection_rule": {
                "any_intervention": "any(block|ask_user|sanitize) across T0/T1/T2 "
                "(symmetric on attack and benign)",
                "block_only": "any(block|ask_user) across T0/T1/T2",
                "primary": "any_intervention",
            },
        }


@dataclass
class TraceReplay:
    decisions: List[GuardDecision]
    total_latency_ms: float
    escalated: bool
    intervened: bool
    blocked: bool
    first_intervention_checkpoint: Optional[str]
    first_block_checkpoint: Optional[str]


def _record_checkpoint(stats: Dict[str, CheckpointStats], decision: GuardDecision) -> None:
    key = decision.checkpoint or "unknown"
    bucket = stats.setdefault(key, CheckpointStats())
    bucket.n += 1
    if _is_hard_block(decision.decision):
        bucket.blocks += 1
    elif decision.decision == Decision.SANITIZE:
        bucket.sanitizes += 1
    else:
        bucket.allows += 1


def _run_trace(guard: Any, trace: TraceStep) -> TraceReplay:
    """
    Replay T0 → T1 → T2 and collect every decision.

    Continues after T0 block so later stages can still be measured when a
    call is present; first_* records the earliest hit of each kind.
    """
    guard.reset_session(keep_registry=False)
    guard.set_user_task(trace.user_request)
    total_ms = 0.0
    escalated = False
    decisions: List[GuardDecision] = []
    first_intervention: Optional[str] = None
    first_block: Optional[str] = None

    def _note(d: GuardDecision) -> None:
        nonlocal total_ms, escalated, first_intervention, first_block
        decisions.append(d)
        total_ms += d.latency_ms
        escalated = escalated or d.escalated
        if first_intervention is None and _is_intervention(d.decision):
            first_intervention = d.checkpoint
        if first_block is None and _is_hard_block(d.decision):
            first_block = d.checkpoint

    if trace.tools:
        _note(guard.check_tool_registration(trace.tools))

    if trace.tools_after:
        _note(guard.check_tool_registration(trace.tools_after))

    for out in trace.tool_outputs:
        _note(
            guard.on_tool_output(
                str(out.get("source", "tool")), str(out.get("content", ""))
            )
        )

    if trace.call is not None:
        _note(guard.check_tool_call(trace.call))

    if not decisions:
        decisions.append(
            GuardDecision(
                decision=Decision.ALLOW,
                risk_score=0.0,
                stage="cascade",
                checkpoint=Checkpoint.T2.value,
                latency_ms=0.0,
            )
        )

    return TraceReplay(
        decisions=decisions,
        total_latency_ms=total_ms,
        escalated=escalated,
        intervened=first_intervention is not None,
        blocked=first_block is not None,
        first_intervention_checkpoint=first_intervention,
        first_block_checkpoint=first_block,
    )


def _is_attack_label(label: str) -> bool:
    return label.lower() in (
        "attack",
        "malicious",
        "adversarial",
        "1",
        "true",
    )


def _update_confusion(c: Confusion, *, is_attack: bool, positive: bool) -> None:
    if is_attack:
        if positive:
            c.tp += 1
        else:
            c.fn += 1
    else:
        if positive:
            c.fp += 1
        else:
            c.tn += 1


def _llm_calls_from_decisions(decisions: Sequence[GuardDecision]) -> int:
    total = 0
    for d in decisions:
        meta = d.meta or {}
        total += int(meta.get("llm_calls") or 0)
    return total


def evaluate_traces(
    traces: Sequence[TraceStep],
    guard: Any = None,
    *,
    policy: Optional[SentinelPolicy] = None,
    collect_records: bool = False,
):
    """
    Run a GuardLike (Sentinel or baseline) on each trace.

    Reports **two** symmetric metrics:
    - ``block_only``: ``block`` / ``ask_user`` only
    - ``any_intervention``: ``block`` / ``ask_user`` / ``sanitize``

    Sanitize on benign counts as a false positive under ``any_intervention``
    (symmetric with counting sanitize as detection on attacks). Primary
    top-level ``tpr``/``fpr`` mirror ``any_intervention``. Also records
    ``auroc`` (risk_score ranking) and ``llm_calls_per_case``.

    When ``collect_records`` is True, returns ``(EvalMetrics, list[dict])``
    with per-trace decisions (for ``results.jsonl.gz`` artefacts).
    """
    g = guard or build_sentinel(policy=policy)
    metrics = EvalMetrics()
    records: List[Dict[str, Any]] = []

    for idx, trace in enumerate(traces):
        # Skip completely empty records.
        if (
            not trace.tools
            and not trace.tools_after
            and not trace.tool_outputs
            and trace.call is None
        ):
            continue

        replay = _run_trace(g, trace)
        for d in replay.decisions:
            _record_checkpoint(metrics.by_checkpoint, d)

        metrics.n += 1
        metrics.latencies_ms.append(replay.total_latency_ms)
        case_llm = _llm_calls_from_decisions(replay.decisions)
        # Also count provider-side counters when present (Task-Shield).
        case_llm = max(case_llm, int(getattr(g, "llm_calls", 0) or 0))
        if hasattr(g, "llm_calls"):
            try:
                g.llm_calls = 0
            except Exception:
                pass
        metrics.llm_calls_total += case_llm
        if replay.escalated:
            metrics.escalations += 1

        fd = replay.first_intervention_checkpoint or "none"
        metrics.first_detection[fd] = metrics.first_detection.get(fd, 0) + 1
        fb = replay.first_block_checkpoint or "none"
        metrics.first_block[fb] = metrics.first_block.get(fb, 0) + 1

        at = trace.attack_type or "other"
        bucket = metrics.by_attack_type.setdefault(
            at, {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )
        bucket["n"] += 1

        is_attack = _is_attack_label(trace.label)
        if is_attack:
            metrics.n_attack += 1
        else:
            metrics.n_benign += 1

        final = replay.decisions[-1]
        metrics.risk_scores.append(float(final.risk_score))
        metrics.attack_labels.append(bool(is_attack))

        _update_confusion(
            metrics.block_only, is_attack=is_attack, positive=replay.blocked
        )
        _update_confusion(
            metrics.any_intervention,
            is_attack=is_attack,
            positive=replay.intervened,
        )

        # Primary / by_attack_type use any-intervention (symmetric).
        if is_attack:
            if replay.intervened:
                metrics.tp += 1
                bucket["tp"] += 1
            else:
                metrics.fn += 1
                bucket["fn"] += 1
        else:
            if replay.intervened:
                metrics.fp += 1
                bucket["fp"] += 1
            else:
                metrics.tn += 1
                bucket["tn"] += 1

        if collect_records:
            trace_id = getattr(trace, "id", None) or f"trace-{idx}"
            rec: Dict[str, Any] = {
                "id": str(trace_id),
                "trace_id": str(trace_id),
                "label": trace.label,
                "is_attack": is_attack,
                "attack_type": at,
                "decision": final.decision.value,
                "blocked": replay.blocked,
                "intervened": replay.intervened,
                "risk_score": final.risk_score,
                "stage": final.stage,
                "checkpoint": final.checkpoint,
                "first_detection": replay.first_intervention_checkpoint,
                "first_block": replay.first_block_checkpoint,
                "escalated": replay.escalated,
                "latency_ms": replay.total_latency_ms,
                "llm_calls": case_llm,
                "reasons": list(final.reasons or []),
                "decisions": [d.to_dict() for d in replay.decisions],
            }
            records.append(rec)

    if metrics.n_attack:
        metrics.asr_without_guard = 1.0
        # ASR with guard: attacks that were *not* hard-blocked (sanitize alone
        # may still leave a residual risk; use block_only for ASR).
        metrics.asr_with_guard = metrics.block_only.fn / metrics.n_attack
    else:
        metrics.asr_without_guard = 0.0
        metrics.asr_with_guard = 0.0

    metrics.auroc = compute_auroc(metrics.risk_scores, metrics.attack_labels)

    if collect_records:
        return metrics, records
    return metrics


def format_metrics(metrics: EvalMetrics, *, title: str = "guard-eval") -> str:
    d = metrics.to_dict()

    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{100.0 * x:.1f}%"

    def ms(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.2f} ms"

    bo = d["block_only"]
    ai = d["any_intervention"]
    lines = [
        f"=== {title} ===",
        f"n={d['n']} (attack={d['n_attack']}, benign={d['n_benign']})",
        "Scoring: symmetric any-intervention (block|ask_user|sanitize) + block-only",
        f"any-intervention TPR={pct(ai['tpr'])}  FPR={pct(ai['fpr'])}  F1={pct(ai['f1'])}",
        f"block-only       TPR={pct(bo['tpr'])}  FPR={pct(bo['fpr'])}  F1={pct(bo['f1'])}",
        f"AUROC={d['auroc'] if d.get('auroc') is not None else 'n/a'}",
        f"ASR without guard={pct(d['asr_without_guard'])}  "
        f"with guard={pct(d['asr_with_guard'])}  "
        f"reduction={pct(d['asr_reduction'])}",
        f"Escalation rate={pct(d['escalation_rate'])}",
        f"Latency mean={ms(d['latency_mean_ms'])}  "
        f"p50={ms(d['latency_p50_ms'])}  p95={ms(d['latency_p95_ms'])}",
        f"LLM calls/case="
        f"{d['llm_calls_per_case'] if d.get('llm_calls_per_case') is not None else 'n/a'}",
        f"any-intervention confusion: TP={ai['tp']} FP={ai['fp']} "
        f"TN={ai['tn']} FN={ai['fn']}",
        f"block-only confusion:       TP={bo['tp']} FP={bo['fp']} "
        f"TN={bo['tn']} FN={bo['fn']}",
    ]
    if d["first_detection"]:
        parts = [f"{k}={v}" for k, v in sorted(d["first_detection"].items())]
        lines.append("First intervention checkpoint: " + ", ".join(parts))
    if d["first_block"]:
        parts = [f"{k}={v}" for k, v in sorted(d["first_block"].items())]
        lines.append("First block checkpoint: " + ", ".join(parts))
    if d["by_checkpoint"]:
        lines.append("By checkpoint:")
        for name, b in sorted(d["by_checkpoint"].items()):
            lines.append(
                f"  {name}: n={b['n']} blocks={b['blocks']} "
                f"sanitize={b['sanitizes']} allow={b['allows']}"
            )
    if d["by_attack_type"]:
        lines.append("By attack_type:")
        for name, b in sorted(d["by_attack_type"].items()):
            lines.append(
                f"  {name}: n={b['n']} TP={b['tp']} FP={b['fp']} "
                f"TN={b['tn']} FN={b['fn']}"
            )
    return "\n".join(lines)
