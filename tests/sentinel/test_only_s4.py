"""only_s4 ablation must invoke S4 (not silently ALLOW when prior risk is 0)."""

from __future__ import annotations

from llm_tester.sentinel.cascade import decide_from_risk
from llm_tester.sentinel.guard import SentinelGuard
from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.types import Decision, StageScore, ToolCall


class RecordingJudge:
    """Mock S4 judge — records calls; returns a fixed score."""

    available = True

    def __init__(self, *, score: float = 0.95, reasons=None) -> None:
        self.score = score
        self.reasons = list(reasons or ["s4:block"])
        self.calls = 0
        self.last_kwargs = None

    def judge(self, user_request, call, snippets, prior_risk, prior_reasons):
        self.calls += 1
        self.last_kwargs = {
            "user_request": user_request,
            "call": call,
            "snippets": snippets,
            "prior_risk": prior_risk,
            "prior_reasons": prior_reasons,
        }
        return StageScore(
            stage="S4",
            score=self.score,
            reasons=list(self.reasons),
            details={"llm_calls": 1, "mock": True},
        )


def test_with_ablation_sets_only_s4_flag():
    p = SentinelPolicy().with_ablation(only_s4=True)
    assert p.only_s4 is True
    assert p.enable_s4 is True
    assert not p.enable_s1 and not p.enable_s2 and not p.enable_s3
    # Other ablations clear the flag.
    p2 = p.with_ablation(no_s2=True)
    assert p2.only_s4 is False


def test_decide_from_risk_only_s4_escalates_at_zero_risk():
    policy = SentinelPolicy().with_ablation(only_s4=True)
    # Empty S1–S3 → risk 0 ≤ tau_low; must still escalate.
    assert decide_from_risk(0.0, policy) == Decision.ESCALATE
    assert decide_from_risk(0.1, policy) == Decision.ESCALATE


def test_decide_from_risk_only_s4_uses_tau_high_on_verdict():
    policy = SentinelPolicy(tau_high=0.7).with_ablation(only_s4=True)
    assert (
        decide_from_risk(0.0, policy, has_s4_verdict=True, s4_score=0.95)
        == Decision.BLOCK
    )
    assert (
        decide_from_risk(0.0, policy, has_s4_verdict=True, s4_score=0.05)
        == Decision.ALLOW
    )
    # Score in (0.5, tau_high) would BLOCK under the old 0.5 rule but ALLOW
    # under only_s4 τ_high=0.7 — documents the ablation-specific threshold.
    assert (
        decide_from_risk(0.0, policy, has_s4_verdict=True, s4_score=0.6)
        == Decision.ALLOW
    )
    assert (
        decide_from_risk(0.0, policy, has_s4_verdict=True, s4_score=0.7)
        == Decision.BLOCK
    )


def test_full_cascade_still_allows_below_tau_low_without_s4():
    """Regression: full cascade / non-only_s4 behaviour unchanged."""
    policy = SentinelPolicy(enable_s4=False, tau_low=0.25, tau_high=0.7)
    assert decide_from_risk(0.0, policy) == Decision.ALLOW
    assert decide_from_risk(0.1, policy) == Decision.ALLOW
    # Grey zone without S4 → allow.
    assert decide_from_risk(0.4, policy) == Decision.ALLOW
    # High risk still blocks.
    assert decide_from_risk(0.8, policy) == Decision.BLOCK


def test_full_cascade_s4_verdict_still_uses_half_threshold():
    policy = SentinelPolicy(enable_s4=True, only_s4=False, tau_high=0.7)
    # Binary mid-band: still BLOCK at ≥0.5 (not τ_high).
    assert (
        decide_from_risk(0.4, policy, has_s4_verdict=True, s4_score=0.6)
        == Decision.BLOCK
    )


def test_only_s4_guard_invokes_judge_and_blocks_on_high_score():
    policy = SentinelPolicy(tau_high=0.7).with_ablation(only_s4=True)
    judge = RecordingJudge(score=0.95, reasons=["s4:block"])
    g = SentinelGuard(policy)
    g.judge = judge
    g.set_user_task("summarise my notes")
    d = g.check_tool_call(
        ToolCall(name="bank.transfer", arguments={"iban": "DE00", "amount": "9"})
    )
    assert judge.calls == 1
    assert d.escalated is True
    assert d.decision == Decision.BLOCK
    assert any(s.stage == "S4" for s in d.stage_scores)
    assert "s4:block" in d.reasons


def test_only_s4_guard_allows_on_low_s4_score():
    policy = SentinelPolicy(tau_high=0.7).with_ablation(only_s4=True)
    judge = RecordingJudge(score=0.05, reasons=["s4:allow"])
    g = SentinelGuard(policy)
    g.judge = judge
    g.set_user_task("summarise my notes")
    d = g.check_tool_call(
        ToolCall(name="notes.read", arguments={"note_id": "1"})
    )
    assert judge.calls == 1
    assert d.escalated is True
    assert d.decision == Decision.ALLOW


def test_only_s4_unavailable_judge_fail_closed():
    policy = SentinelPolicy().with_ablation(only_s4=True)
    g = SentinelGuard(policy)
    g.judge = None  # enable_s4 True but judge missing
    g.set_user_task("do something")
    d = g.check_tool_call(ToolCall(name="shell.exec", arguments={"cmd": "rm -rf /"}))
    assert d.decision == Decision.BLOCK
    assert "s4:unavailable_fail_closed" in d.reasons
