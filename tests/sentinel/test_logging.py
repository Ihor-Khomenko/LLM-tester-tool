"""JSONL decision logger tests."""

from pathlib import Path

from llm_tester.sentinel.guard import SentinelGuard
from llm_tester.sentinel.logging_util import DecisionLogger
from llm_tester.sentinel.types import Decision, GuardDecision, ToolCall


def test_decision_logger_writes_jsonl(tmp_path: Path):
    path = tmp_path / "dec.jsonl"
    with DecisionLogger(path) as log:
        d = GuardDecision(
            decision=Decision.BLOCK,
            risk_score=0.9,
            stage="S2",
            checkpoint="T2",
            reasons=["s2:test"],
            latency_ms=1.2,
        )
        log.log(d, trace_id="t1")
    text = path.read_text(encoding="utf-8")
    assert "block" in text
    assert "trace_id" in text


def test_guard_logs_when_path_set(tmp_path: Path):
    path = tmp_path / "g.jsonl"
    g = SentinelGuard(log_path=path)
    g.check_tool_call(ToolCall(name="noop", arguments={}), user_request="hi")
    g.logger.close()
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip()
