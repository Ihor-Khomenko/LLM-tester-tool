"""Risk aggregation and cascade decision logic."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.types import Decision, StageScore


def aggregate_risk(
    scores: Sequence[StageScore],
    policy: SentinelPolicy,
) -> Tuple[float, List[str]]:
    """Weighted max/mean hybrid of enabled stage scores → r in [0, 1]."""
    if not scores:
        return 0.0, []

    weights = {
        "S1": policy.weight_s1 if policy.enable_s1 else 0.0,
        "S2": policy.weight_s2 if policy.enable_s2 else 0.0,
        "S3": policy.weight_s3 if policy.enable_s3 else 0.0,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    peak = 0.0
    reasons: List[str] = []

    for s in scores:
        w = weights.get(s.stage, 0.0)
        if w <= 0 and s.stage in weights:
            continue
        if s.stage == "S4":
            # S4 overrides when present.
            return float(min(1.0, s.score)), list(s.reasons)
        peak = max(peak, s.score)
        if w > 0:
            weighted_sum += w * s.score
            weight_total += w
        reasons.extend(s.reasons)

    if weight_total <= 0:
        return float(min(1.0, peak)), reasons

    mean = weighted_sum / weight_total
    # Emphasize peaks so a single strong S2/S3 hit is not diluted by zeros.
    risk = 0.45 * mean + 0.55 * peak
    # Hard floor: a near-certain stage score must dominate the cascade.
    if peak >= 0.85:
        risk = max(risk, peak)
    return float(min(1.0, max(0.0, risk))), reasons


def decide_from_risk(
    risk: float,
    policy: SentinelPolicy,
    *,
    has_s4_verdict: bool = False,
    s4_score: Optional[float] = None,
    stage_scores: Optional[Sequence[StageScore]] = None,
) -> Decision:
    if has_s4_verdict and s4_score is not None:
        if s4_score >= 0.5:
            return (
                Decision.ASK_USER
                if policy.ask_user_on_block or policy.high_risk_action == "ask_user"
                else Decision.BLOCK
            )
        return Decision.ALLOW

    # Per-stage hard thresholds (S3 critical mismatch → block/escalate).
    if stage_scores:
        for s in stage_scores:
            if s.stage == "S3" and s.score >= policy.s3_block_threshold:
                if policy.enable_s4:
                    return Decision.ESCALATE
                return (
                    Decision.ASK_USER
                    if policy.ask_user_on_block or policy.high_risk_action == "ask_user"
                    else Decision.BLOCK
                )
            if s.stage == "S2" and s.score >= policy.s2_block_threshold:
                if risk < policy.tau_high:
                    risk = max(risk, s.score)
            if s.stage == "S1" and s.score >= policy.s1_block_threshold:
                risk = max(risk, s.score)

    if risk >= policy.tau_high:
        if policy.high_risk_action == "ask_user" or policy.ask_user_on_block:
            return Decision.ASK_USER
        return Decision.BLOCK
    if risk <= policy.tau_low:
        return Decision.ALLOW
    # Grey zone: escalate to S4 when enabled.
    if policy.enable_s4:
        return Decision.ESCALATE
    # Without S4: block when a stage already hit a hard threshold (handled
    # above) or risk is in the upper half of the grey band; otherwise allow
    # (preserves utility). Critical S3 on side-effect tools scores ≥0.85 and
    # clears tau_high via the peak floor.
    mid = (policy.tau_low + policy.tau_high) / 2.0
    if risk >= mid:
        return (
            Decision.ASK_USER
            if policy.ask_user_on_block or policy.high_risk_action == "ask_user"
            else Decision.BLOCK
        )
    return Decision.ALLOW
