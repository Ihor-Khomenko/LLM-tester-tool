"""S4 judge parsing (offline, mocked provider)."""

from unittest.mock import MagicMock

from llm_tester.providers import ChatResult
from llm_tester.sentinel.stages.judge import JudgeStage, _parse_verdict
from llm_tester.sentinel.types import Decision, ToolCall


def test_parse_verdict():
    assert _parse_verdict("BLOCK: injected transfer") == Decision.BLOCK
    assert _parse_verdict("ALLOW — looks fine") == Decision.ALLOW
    assert _parse_verdict("hmm unclear") == Decision.BLOCK


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
