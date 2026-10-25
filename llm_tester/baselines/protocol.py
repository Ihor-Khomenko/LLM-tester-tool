"""Duck-typed guard surface shared by MCP-Sentinel and paper baselines."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union, runtime_checkable

from llm_tester.sentinel.types import (
    Checkpoint,
    Decision,
    GuardDecision,
    ToolCall,
    ToolSpec,
)


@runtime_checkable
class GuardLike(Protocol):
    """Minimal API consumed by ``evaluate_traces`` / agent-bench / AgentDojo."""

    def set_user_task(self, text: str) -> None: ...

    def reset_session(self, *, keep_registry: bool = True) -> None: ...

    def check_tool_registration(
        self, tools: Sequence[Union[ToolSpec, Dict[str, Any]]]
    ) -> GuardDecision: ...

    def on_tool_output(
        self, source: str, content: str, *, sanitize: bool = True
    ) -> GuardDecision: ...

    def check_tool_call(
        self,
        call: Union[ToolCall, Dict[str, Any]],
        *,
        user_request: Optional[str] = None,
    ) -> GuardDecision: ...


def allow_decision(
    *,
    checkpoint: str,
    stage: str = "baseline",
    risk: float = 0.0,
    latency_ms: float = 0.0,
    reasons: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    sanitized_content: Optional[str] = None,
    decision: Decision = Decision.ALLOW,
) -> GuardDecision:
    return GuardDecision(
        decision=decision,
        risk_score=float(min(1.0, max(0.0, risk))),
        stage=stage,
        checkpoint=checkpoint,
        reasons=list(reasons or []),
        latency_ms=latency_ms,
        sanitized_content=sanitized_content,
        meta=dict(meta or {}),
    )


def as_tool_call(call: Union[ToolCall, Dict[str, Any]]) -> ToolCall:
    if isinstance(call, ToolCall):
        return call
    return ToolCall(
        name=str(call.get("name", "")),
        arguments=dict(call.get("arguments") or call.get("args") or {}),
    )


class BaseGuard:
    """Shared session state for baseline guards."""

    name: str = "baseline"

    def __init__(self) -> None:
        self.user_request: str = ""
        self.outputs: List[Dict[str, str]] = []
        self.registered: List[str] = []
        self.llm_calls: int = 0
        self.available: bool = True
        self.skip_reason: Optional[str] = None

    def set_user_task(self, text: str) -> None:
        self.user_request = text or ""

    def reset_session(self, *, keep_registry: bool = True) -> None:
        self.outputs.clear()
        self.llm_calls = 0
        if not keep_registry:
            self.registered.clear()

    def check_tool_registration(
        self, tools: Sequence[Union[ToolSpec, Dict[str, Any]]]
    ) -> GuardDecision:
        start = time.perf_counter()
        for raw in tools:
            name = (
                raw.name
                if isinstance(raw, ToolSpec)
                else str(raw.get("name", ""))
            )
            if name and name not in self.registered:
                self.registered.append(name)
        return allow_decision(
            checkpoint=Checkpoint.T0.value,
            stage=self.name,
            latency_ms=(time.perf_counter() - start) * 1000,
            meta={"n_tools": len(list(tools)), "llm_calls": 0},
        )

    def _note_output(self, source: str, content: str) -> None:
        self.outputs.append({"source": source, "content": content or ""})
