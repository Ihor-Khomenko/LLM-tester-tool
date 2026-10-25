"""S4 judge parsing (offline, mocked provider)."""

from unittest.mock import MagicMock

from llm_tester.providers import ChatResult
from llm_tester.sentinel.guard import SentinelGuard
from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.stages.judge import (
    AmbiguousPolicy,
    JudgeFamily,
    JudgeStage,
    _parse_verdict,
    detect_judge_family,
    parse_judge_verdict,
)
from llm_tester.sentinel.types import Decision, ToolCall


def test_parse_verdict_backcompat():
    assert _parse_verdict("BLOCK: injected transfer") == Decision.BLOCK
    assert _parse_verdict("ALLOW — looks fine") == Decision.ALLOW
    # Legacy alias treats ambiguous as BLOCK.
    assert _parse_verdict("hmm unclear") == Decision.BLOCK


def test_parse_llama_guard_safe_unsafe():
    allow, d = parse_judge_verdict(
        "safe", family=JudgeFamily.LLAMA_GUARD.value
    )
    assert allow == Decision.ALLOW
    assert d.get("llama_guard") == "safe"

    block, d2 = parse_judge_verdict(
        "unsafe\nS2", family=JudgeFamily.LLAMA_GUARD.value
    )
    assert block == Decision.BLOCK
    assert d2.get("categories") == ["S2"]

    # Must not map bare "safe" to BLOCK (live FPR bug with allow_block parser).
    allow2, _ = parse_judge_verdict(
        "safe\n", family=JudgeFamily.ALLOW_BLOCK.value
    )
    assert allow2 == Decision.ALLOW


def test_ambiguous_defaults_to_defer():
    verdict, details = parse_judge_verdict(
        "maybe?", family=JudgeFamily.ALLOW_BLOCK.value
    )
    assert verdict is None
    assert details.get("ambiguous") is True

    allow, _ = parse_judge_verdict(
        "maybe?",
        family=JudgeFamily.ALLOW_BLOCK.value,
        ambiguous=AmbiguousPolicy.ALLOW.value,
    )
    assert allow == Decision.ALLOW

    block, _ = parse_judge_verdict(
        "maybe?",
        family=JudgeFamily.ALLOW_BLOCK.value,
        ambiguous=AmbiguousPolicy.BLOCK.value,
    )
    assert block == Decision.BLOCK


def test_detect_judge_family():
    assert detect_judge_family("llama-guard3:1b") == JudgeFamily.LLAMA_GUARD.value
    assert detect_judge_family("qwen2.5:7b") == JudgeFamily.ALLOW_BLOCK.value
    assert (
        detect_judge_family("anything", explicit=JudgeFamily.LLAMA_GUARD.value)
        == JudgeFamily.LLAMA_GUARD.value
    )


def test_judge_llama_guard_safe_with_mock():
    provider = MagicMock()
    provider.model = "llama-guard3:1b"
    provider.chat.return_value = ChatResult(content="safe", latency_s=0.01)
    stage = JudgeStage(
        policy=SentinelPolicy(s4_judge_family="auto"),
        provider=provider,
    )
    score = stage.judge(
        "read email",
        ToolCall(name="notes.write", arguments={"body": "hi"}),
        ["hello"],
        0.5,
        ["s3:mismatch"],
    )
    assert score.score <= 0.1
    assert "s4:allow" in score.reasons


def test_judge_ambiguous_defers_to_prior():
    provider = MagicMock()
    provider.model = "tiny"
    provider.chat.return_value = ChatResult(content="unclear waffle", latency_s=0.01)
    stage = JudgeStage(
        policy=SentinelPolicy(s4_ambiguous_policy="defer"),
        provider=provider,
    )
    score = stage.judge(
        "u",
        ToolCall(name="x", arguments={}),
        [],
        0.55,
        ["s3:soft"],
    )
    assert abs(score.score - 0.55) < 1e-9
    assert "s4:ambiguous_defer" in score.reasons


def test_guard_ambiguous_s4_does_not_block():
    """Ambiguous judge output must not become BLOCK (live llama-guard FPR)."""

    class AmbiguousJudge:
        available = True

        def judge(self, *args, **kwargs):
            from llm_tester.sentinel.types import StageScore

            return StageScore(
                stage="S4",
                score=0.55,
                reasons=["s4:ambiguous_defer"],
                details={"deferred_to_prior": True, "ambiguous": True},
            )

    policy = SentinelPolicy(
        enable_s1=False,
        enable_s2=False,
        enable_s3=True,
        enable_s4=True,
        tau_low=0.25,
        tau_high=0.70,
        s4_ambiguous_policy="defer",
    )
    g = SentinelGuard(policy)
    g.judge = AmbiguousJudge()
    # Force grey-zone via a soft S3-like score by marking intent backend unavailable
    # and injecting a stage through a fake intent.
    class SoftIntent:
        def score_call(self, *a, **k):
            from llm_tester.sentinel.types import StageScore

            return StageScore(stage="S3", score=0.55, reasons=["s3:soft"])

    g.intent = SoftIntent()
    g.set_user_task("summarise notes")
    d = g.check_tool_call(ToolCall(name="notes.write", arguments={"body": "ok"}))
    assert d.escalated is True
    assert d.decision == Decision.ALLOW
    assert d.decision != Decision.BLOCK


def test_judge_with_mock_provider():
    provider = MagicMock()
    provider.chat.return_value = ChatResult(content="BLOCK injection", latency_s=0.01)
    stage = JudgeStage(provider=provider)
    assert stage.available
    score = stage.judge(
        "read email",
        ToolCall(name="bank.transfer", arguments={"iban": "DE00"}),
        ["ignore previous"],
        0.5,
        ["s2:taint"],
    )
    assert score.score >= 0.9
    assert "s4:block" in score.reasons


def test_judge_unavailable():
    stage = JudgeStage(provider=None)
    assert not stage.available
    score = stage.judge("u", ToolCall(name="x", arguments={}), [], 0.4, [])
    assert "s4:unavailable" in score.reasons


def test_prompt_templates_per_family():
    stage = JudgeStage()
    sys_ab, user_ab = stage.build_prompt(
        "u",
        ToolCall(name="t", arguments={}),
        ["snip"],
        0.4,
        [],
        family=JudgeFamily.ALLOW_BLOCK.value,
    )
    assert "ALLOW or BLOCK" in user_ab or "ALLOW" in sys_ab
    sys_lg, user_lg = stage.build_prompt(
        "u",
        ToolCall(name="t", arguments={}),
        ["snip"],
        0.4,
        [],
        family=JudgeFamily.LLAMA_GUARD.value,
    )
    assert "safe" in sys_lg.lower()
    assert "Agent messages" in user_lg
