"""Offline evaluation of MCP-Sentinel on traces (imports sentinel; not part of it)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from llm_tester.agent.traces import TraceStep
from llm_tester.sentinel.guard import SentinelGuard, build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.types import Decision


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


@dataclass
class EvalMetrics:
    n: int = 0
    n_attack: int = 0
    n_benign: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    escalations: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    asr_without_guard: float = 1.0
    asr_with_guard: float = 0.0
    by_attack_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)

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
            "asr_without_guard": self.asr_without_guard,
            "asr_with_guard": self.asr_with_guard,
            "asr_reduction": self.asr_reduction,
            "escalation_rate": self.escalation_rate,
            "escalations": self.escalations,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "by_attack_type": self.by_attack_type,
            # Per-trace records are written to results.jsonl.gz, not metrics.json.
        }


def _is_block(decision: Decision) -> bool:
    return decision in (Decision.BLOCK, Decision.ASK_USER)


def _run_trace(guard: SentinelGuard, trace: TraceStep) -> tuple:
    """
    Replay one trace through T0 → T1 → T2.

    Returns (final_decision, total_latency_ms, escalated).
    For rug_pull / tool_poisoning, T0 may already decide.
    """
    guard.reset_session(keep_registry=False)
    guard.set_user_task(trace.user_request)
    total_ms = 0.0
    escalated = False
    last = None

    if trace.tools:
        last = guard.check_tool_registration(trace.tools)
        total_ms += last.latency_ms
        escalated = escalated or last.escalated
        if _is_block(last.decision) and not trace.tools_after and trace.call is None:
            return last, total_ms, escalated

    if trace.tools_after:
        # Rug pull: second tools/list observation after pin.
        last = guard.check_tool_registration(trace.tools_after)
        total_ms += last.latency_ms
        escalated = escalated or last.escalated
        if _is_block(last.decision) and trace.call is None:
            return last, total_ms, escalated

    for out in trace.tool_outputs:
        t1 = guard.on_tool_output(
            str(out.get("source", "tool")), str(out.get("content", ""))
        )
        total_ms += t1.latency_ms
        last = t1

    if trace.call is not None:
        last = guard.check_tool_call(trace.call)
        total_ms += last.latency_ms
        escalated = escalated or last.escalated

    if last is None:
        # Empty trace — treat as allow with zero risk.
        from llm_tester.sentinel.types import Checkpoint, GuardDecision

        last = GuardDecision(
            decision=Decision.ALLOW,
            risk_score=0.0,
            stage="cascade",
            checkpoint=Checkpoint.T2.value,
            latency_ms=0.0,
        )
    return last, total_ms, escalated


def evaluate_traces(
    traces: Sequence[TraceStep],
    guard: Optional[SentinelGuard] = None,
    *,
    policy: Optional[SentinelPolicy] = None,
    collect_records: bool = False,
) -> EvalMetrics:
    """
    Run the cascade on each trace.

    Attack label → predicted block = TP; predicted allow = FN.
    Benign label → predicted allow = TN; predicted block = FP.

    When ``collect_records`` is True, per-trace dicts are stored on
    ``metrics.records`` for run-artifact JSONL output.
    """
    g = guard or build_sentinel(policy=policy)
    metrics = EvalMetrics()
    records: List[Dict[str, Any]] = []

    for trace in traces:
        # Skip completely empty records.
        if (
            not trace.tools
            and not trace.tools_after
            and not trace.tool_outputs
            and trace.call is None
        ):
            continue

        decision, latency_ms, escalated = _run_trace(g, trace)
        blocked = _is_block(decision.decision)
        metrics.n += 1
        metrics.latencies_ms.append(latency_ms)
        if escalated:
            metrics.escalations += 1

        at = trace.attack_type or "other"
        bucket = metrics.by_attack_type.setdefault(
            at, {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )
        bucket["n"] += 1

        is_attack = trace.label.lower() in (
            "attack",
            "malicious",
            "adversarial",
            "1",
            "true",
        )
        outcome = "tn"
        if is_attack:
            metrics.n_attack += 1
            if blocked:
                metrics.tp += 1
                bucket["tp"] += 1
                outcome = "tp"
            else:
                metrics.fn += 1
                bucket["fn"] += 1
                outcome = "fn"
        else:
            metrics.n_benign += 1
            if blocked:
                metrics.fp += 1
                bucket["fp"] += 1
                outcome = "fp"
            else:
                metrics.tn += 1
                bucket["tn"] += 1
                outcome = "tn"

        if collect_records:
            records.append(
                {
                    "id": trace.id,
                    "label": trace.label,
                    "attack_type": trace.attack_type,
                    "blocked": blocked,
                    "outcome": outcome,
                    "escalated": escalated,
                    "latency_ms": latency_ms,
                    "decision": decision.to_dict(),
                }
            )

    if metrics.n_attack:
        metrics.asr_without_guard = 1.0
        metrics.asr_with_guard = metrics.fn / metrics.n_attack
    else:
        metrics.asr_without_guard = 0.0
        metrics.asr_with_guard = 0.0

    if collect_records:
        metrics.records = records
    return metrics


def format_metrics(metrics: EvalMetrics, *, title: str = "guard-eval") -> str:
    d = metrics.to_dict()

    def pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{100.0 * x:.1f}%"

    def ms(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.2f} ms"

    lines = [
        f"=== {title} ===",
        f"n={d['n']} (attack={d['n_attack']}, benign={d['n_benign']})",
        f"TPR={pct(d['tpr'])}  FPR={pct(d['fpr'])}  F1={pct(d['f1'])}",
        f"ASR without guard={pct(d['asr_without_guard'])}  "
        f"with guard={pct(d['asr_with_guard'])}  "
        f"reduction={pct(d['asr_reduction'])}",
        f"Escalation rate={pct(d['escalation_rate'])}",
        f"Latency mean={ms(d['latency_mean_ms'])}  "
        f"p50={ms(d['latency_p50_ms'])}  p95={ms(d['latency_p95_ms'])}",
        f"Confusion: TP={d['tp']} FP={d['fp']} TN={d['tn']} FN={d['fn']}",
    ]
    if d["by_attack_type"]:
        lines.append("By attack_type:")
        for name, b in sorted(d["by_attack_type"].items()):
            lines.append(
                f"  {name}: n={b['n']} TP={b['tp']} FP={b['fp']} "
                f"TN={b['tn']} FN={b['fn']}"
            )
    return "\n".join(lines)
