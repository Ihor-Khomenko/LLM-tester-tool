"""Risk aggregation and cascade decision logic."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.types import Decision, StageScore


def aggregate_risk(
    scores: Sequence[StageScore],
    policy: SentinelPolicy,
) -> Tuple[float, List[str]]:
    """
    Combine **enabled** stage scores into risk r ∈ [0, 1].

    Disabled stages must not appear in ``scores`` (caller skips them). Aggregation
    is monotone in the set of stages: removing a stage cannot raise risk.

    Modes (``policy.risk_aggregation``):
    - ``noisy_or`` (default): ``1 - ∏(1 - s_i)`` — ablating a stage cleanly
      removes its factor; no dilution by zero-weight averages.
    - ``max``: peak of enabled scores.
    - ``weighted``: normalised weighted mean using ``weight_s1/s2/s3`` for
      stages present in ``scores`` (disabled stages omitted — no zero dilution).
    """
    if not scores:
        return 0.0, []

    reasons: List[str] = []
    active: List[float] = []
    weighted_pairs: List[Tuple[str, float]] = []

    for s in scores:
        if s.stage == "S4":
            # S4 verdict replaces the cascade score when present.
            return float(min(1.0, max(0.0, s.score))), list(s.reasons)
        reasons.extend(s.reasons)
        val = float(min(1.0, max(0.0, s.score)))
        active.append(val)
        weighted_pairs.append((s.stage.upper(), val))

    if not active:
        return 0.0, reasons

    mode = (policy.risk_aggregation or "noisy_or").lower().strip()
    if mode == "max":
        risk = max(active)
    elif mode == "weighted":
        weight_map = {
            "S1": float(policy.weight_s1),
            "S2": float(policy.weight_s2),
            "S3": float(policy.weight_s3),
        }
        num = 0.0
        den = 0.0
        for stage, val in weighted_pairs:
            w = weight_map.get(stage, 1.0)
            if w <= 0:
                continue
            num += w * val
            den += w
        risk = (num / den) if den else max(active)
    else:
        # noisy-OR (default)
        surv = 1.0
        for s in active:
            surv *= 1.0 - s
        risk = 1.0 - surv

    return float(min(1.0, max(0.0, risk))), reasons


def _s3_dominant_uncertain(stage_scores: Sequence[StageScore], policy: SentinelPolicy) -> bool:
    """True when risk is driven mainly by soft S3 intent (prefer S4 over hard block)."""
    by = {s.stage: s for s in stage_scores}
    s3 = by.get("S3")
    if s3 is None or s3.score < policy.tau_low:
        return False
    s2 = by.get("S2")
    s1 = by.get("S1")
    s2_score = s2.score if s2 else 0.0
    s1_score = s1.score if s1 else 0.0
    # Strong provenance / signatures already justify a hard decision.
    if s2_score >= policy.s2_block_threshold or s1_score >= policy.s1_block_threshold:
        return False
    return s3.score >= max(s2_score, s1_score)


def decide_from_risk(
    risk: float,
    policy: SentinelPolicy,
    *,
    has_s4_verdict: bool = False,
    s4_score: Optional[float] = None,
    stage_scores: Optional[Sequence[StageScore]] = None,
) -> Decision:
    only_s4 = bool(getattr(policy, "only_s4", False)) and bool(policy.enable_s4)

    if has_s4_verdict and s4_score is not None:
        # Full cascade: binary S4 ALLOW/BLOCK scores use 0.5.
        # only_s4 ablation: S4 score alone vs tau_high (default 0.7).
        threshold = float(policy.tau_high) if only_s4 else 0.5
        if s4_score >= threshold:
            return (
                Decision.ASK_USER
                if policy.ask_user_on_block or policy.high_risk_action == "ask_user"
                else Decision.BLOCK
            )
        return Decision.ALLOW

    # only_s4: S1–S3 are off so prior risk is 0 (≤ tau_low). Still escalate
    # every T2 call to S4 — otherwise the ablation is a no-op.
    if only_s4 and not has_s4_verdict:
        return Decision.ESCALATE

    stages = stage_scores or ()

    if risk >= policy.tau_high:
        # Restore grey-zone path: uncertain S3-lite intent escalates to S4
        # instead of hard-blocking without a judge.
        if policy.enable_s4 and _s3_dominant_uncertain(stages, policy):
            return Decision.ESCALATE
        if policy.high_risk_action == "ask_user" or policy.ask_user_on_block:
            return Decision.ASK_USER
        return Decision.BLOCK

    if risk <= policy.tau_low:
        return Decision.ALLOW

    # Grey zone: the intended S4 path. Without S4, allow (utility / FPR) —
    # calibrate ``tau_high`` rather than mid-splitting the band.
    if policy.enable_s4:
        return Decision.ESCALATE
    return Decision.ALLOW
