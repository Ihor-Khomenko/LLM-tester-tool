"""Harness correctness: guarded AbortAgentError must not leak unexecuted tool_calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from llm_tester.agent.agentdojo_adapter import (
    AgentDojoCaseResult,
    AgentDojoSummary,
    _apply_security_scores,
    _score_attack_attempted,
    messages_without_unexecuted_tool_calls,
    try_import_agentdojo,
)
from llm_tester.agent.agentdojo_fullrun import summarize_with_wilson
from llm_tester.sentinel.types import Decision, GuardDecision, ToolCall


@dataclass
class _FakeTC:
    function: str
    args: Dict[str, Any]


class _BlockNamedGuard:
    """Minimal GuardLike: BLOCK the first matching tool name at T2."""

    def __init__(self, block_name: str) -> None:
        self.block_name = block_name

    def reset_session(self, keep_registry: bool = False) -> None:
        return None

    def set_user_task(self, task: str) -> None:
        return None

    def check_tool_registration(self, tools: Any) -> None:
        return None

    def check_tool_call(self, call: ToolCall) -> GuardDecision:
        if call.name == self.block_name:
            return GuardDecision(
                decision=Decision.BLOCK,
                risk_score=1.0,
                stage="S2",
                checkpoint="T2",
                reasons=[f"test-block:{call.name}"],
            )
        return GuardDecision(
            decision=Decision.ALLOW,
            risk_score=0.0,
            stage="S1",
            checkpoint="T2",
            reasons=["allow"],
        )

    def on_tool_output(self, name: str, text: str) -> GuardDecision:
        return GuardDecision(
            decision=Decision.ALLOW,
            risk_score=0.0,
            stage="S1",
            checkpoint="T1",
            reasons=["allow"],
        )


def test_messages_without_unexecuted_tool_calls_drops_whole_batch():
    """T2 fail-closed: executed_prefix=0 drops blocked + later + earlier peers."""
    batch = [
        _FakeTC("invite_user_to_slack", {"user": "Fred"}),
        _FakeTC("add_user_to_channel", {"user": "Fred", "channel": "general"}),
        _FakeTC("remove_user_from_slack", {"user": "Fred"}),
    ]
    prior = {
        "role": "assistant",
        "content": [],
        "tool_calls": [_FakeTC("get_webpage", {"url": "http://x"})],
    }
    last = {"role": "assistant", "content": [], "tool_calls": batch}
    messages = [
        {"role": "user", "content": "hi"},
        prior,
        {"role": "tool", "content": "ok"},
        last,
    ]
    out = messages_without_unexecuted_tool_calls(messages, executed_prefix=0)
    assert out is not messages
    assert out[-1]["tool_calls"] == []
    # Prior executed turn kept intact.
    assert out[1]["tool_calls"][0].function == "get_webpage"
    # Original untouched.
    assert len(messages[-1]["tool_calls"]) == 3


def test_messages_without_unexecuted_keeps_executed_prefix():
    batch = [_FakeTC("a", {}), _FakeTC("b", {}), _FakeTC("c", {})]
    messages = [{"role": "assistant", "content": [], "tool_calls": batch}]
    out = messages_without_unexecuted_tool_calls(messages, executed_prefix=1)
    assert [tc.function for tc in out[-1]["tool_calls"]] == ["a"]


def test_score_attack_attempted_falls_back_without_transcript():
    inj = MagicMock()
    inj.security_from_traces.return_value = True
    assert _score_attack_attempted(
        inj, attempted_messages=None, executed_security=False
    ) is False


def test_summary_exposes_attempted_vs_executed_asr():
    summary = AgentDojoSummary(n=2, n_attack=2)
    summary.attack_successes = 0
    summary.attack_successes_attempted = 2
    m = summarize_with_wilson(summary)
    assert m["asr"] == 0.0
    assert m["asr_attempted"] == 1.0
    assert m["attack_successes"] == 0
    assert m["attack_successes_attempted"] == 2
    assert "asr_attempted_wilson" in m
    assert "targeted_asr_attempted_wilson" in m


@pytest.mark.skipif(try_import_agentdojo() is None, reason="agentdojo not installed")
def test_guarded_parallel_batch_block_dual_asr_injection_task_5():
    """Block invite in InjectionTask5 batch → executed ASR False, attempted True."""
    from llm_tester.agent.agentdojo_adapter import (
        _register_suite_tools,
        build_agentdojo_llm,
        build_agentdojo_pipeline,
        load_agentdojo_suite,
    )

    suite = load_agentdojo_suite("slack", benchmark_version="v1.2.2")
    user_task = suite.user_tasks["user_task_0"]
    injection_task = suite.injection_tasks["injection_task_5"]

    guard = _BlockNamedGuard("invite_user_to_slack")
    llm, mode = build_agentdojo_llm(provider_name="mock")
    assert mode == "mock"
    pipeline, guarded = build_agentdojo_pipeline(
        llm=llm, defense=guard, defense_name="block_invite"
    )

    guard.reset_session(keep_registry=False)
    guard.set_user_task(str(getattr(user_task, "PROMPT", "") or ""))
    _register_suite_tools(guard, suite)
    llm.reset_case_calls()
    llm.configure(
        user_task=user_task,
        injection_task=injection_task,
        compromised=True,
    )
    guarded.reset_trace()

    utility, security = suite.run_task_with_pipeline(
        pipeline,
        user_task,
        injection_task,
        {},
    )
    assert guarded.blocked is True
    assert any(s.get("blocked") == "invite_user_to_slack" for s in guarded.steps)
    # Executed-trace score from AgentDojo must be False (calls never ran).
    assert security is False

    rec = AgentDojoCaseResult(
        task_id="user_task_0__injection_task_5",
        label="attack",
        defense="block_invite",
        blocked=True,
        attack_success=False,
        utility_success=bool(utility),
    )
    _apply_security_scores(
        rec,
        injection_task=injection_task,
        security=bool(security),
        guarded=guarded,
    )
    assert rec.attack_success is False
    assert rec.attack_success_attempted is True
    assert guarded.attempted_messages_at_abort is not None
    # Stripped abort transcript has no attack tool_calls on the blocked turn.
    attempted_last = guarded.attempted_messages_at_abort[-1]
    assert len(attempted_last.get("tool_calls") or []) >= 3


@pytest.mark.skipif(try_import_agentdojo() is None, reason="agentdojo not installed")
def test_none_arm_unblocked_scoring_unchanged():
    """none arm (no guard): attack tools run; attempted == executed == True."""
    from llm_tester.agent.agentdojo_adapter import (
        build_agentdojo_llm,
        build_agentdojo_pipeline,
        load_agentdojo_suite,
    )

    suite = load_agentdojo_suite("slack", benchmark_version="v1.2.2")
    user_task = suite.user_tasks["user_task_0"]
    injection_task = suite.injection_tasks["injection_task_5"]

    llm, _ = build_agentdojo_llm(provider_name="mock")
    pipeline, guarded = build_agentdojo_pipeline(
        llm=llm, defense=None, defense_name="none"
    )
    llm.configure(
        user_task=user_task,
        injection_task=injection_task,
        compromised=True,
    )
    guarded.reset_trace()

    _utility, security = suite.run_task_with_pipeline(
        pipeline,
        user_task,
        injection_task,
        {},
    )
    assert guarded.blocked is False
    assert security is True

    rec = AgentDojoCaseResult(
        task_id="user_task_0__injection_task_5",
        label="attack",
        defense="none",
        blocked=False,
        attack_success=False,
        utility_success=False,
    )
    _apply_security_scores(
        rec,
        injection_task=injection_task,
        security=bool(security),
        guarded=guarded,
    )
    assert rec.attack_success is True
    assert rec.attack_success_attempted is True
    assert getattr(guarded, "attempted_messages_at_abort", None) is None
