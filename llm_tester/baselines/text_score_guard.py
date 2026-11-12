"""Shared T1/T2 wiring for score-based detector baselines (PromptGuard-style).

Subclasses implement ``score_text`` (and optional lazy model load). Applied at
the same checkpoints as ``prompt_guard2``: tool outputs (T1 → SANITIZE) and
tool-call verbalizations (T2 → BLOCK). T0 inherits BaseGuard (register only).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

from llm_tester.baselines.protocol import BaseGuard, allow_decision, as_tool_call
from llm_tester.sentinel.types import Checkpoint, Decision, GuardDecision, ToolCall


class TextScoreGuard(BaseGuard):
    """Classify tool outputs / call verbalizations via ``score_text``."""

    name = "text_score"
    skip_reason_tag: str = "skipped"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        classifier: Any = None,
    ) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.available = False
        self.skip_reason: Optional[str] = None
        self._classifier_fn = classifier
        if classifier is not None:
            self.available = True

    def score_text(self, text: str) -> float:
        if not self.available:
            return 0.0
        if self._classifier_fn is not None:
            return float(min(1.0, max(0.0, self._classifier_fn(text or ""))))
        raise NotImplementedError

    def _skip_decision(self, checkpoint: str, start: float) -> GuardDecision:
        return allow_decision(
            checkpoint=checkpoint,
            stage=self.name,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=[f"{self.name}:{self.skip_reason_tag}"],
            meta={
                "available": False,
                "skip_reason": self.skip_reason,
                "llm_calls": 0,
            },
        )

    def on_tool_output(
        self, source: str, content: str, *, sanitize: bool = True
    ) -> GuardDecision:
        start = time.perf_counter()
        self._note_output(source, content or "")
        if not self.available:
            return self._skip_decision(Checkpoint.T1.value, start)
        score = self.score_text(content or "")
        decision = Decision.SANITIZE if score >= self.threshold else Decision.ALLOW
        reasons = [f"{self.name}:score={score:.3f}"]
        if decision == Decision.SANITIZE:
            reasons.append(f"{self.name}:injection")
        return allow_decision(
            checkpoint=Checkpoint.T1.value,
            stage=self.name,
            risk=score,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=reasons,
            decision=decision,
            sanitized_content=(content or "") if decision == Decision.SANITIZE else None,
            meta={"available": True, "score": score, "llm_calls": 0},
        )

    def check_tool_call(
        self,
        call: Union[ToolCall, Dict[str, Any]],
        *,
        user_request: Optional[str] = None,
    ) -> GuardDecision:
        start = time.perf_counter()
        if user_request is not None:
            self.set_user_task(user_request)
        tc = as_tool_call(call)
        if not self.available:
            return self._skip_decision(Checkpoint.T2.value, start)
        blob = f"User: {self.user_request}\nCall: {tc.verbalize()}"
        for o in self.outputs[-3:]:
            blob += f"\nOutput[{o['source']}]: {o['content'][:500]}"
        score = self.score_text(blob)
        decision = Decision.BLOCK if score >= self.threshold else Decision.ALLOW
        return allow_decision(
            checkpoint=Checkpoint.T2.value,
            stage=self.name,
            risk=score,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=[
                f"{self.name}:score={score:.3f}",
                f"{self.name}:block" if decision == Decision.BLOCK else f"{self.name}:allow",
            ],
            decision=decision,
            meta={"available": True, "score": score, "llm_calls": 0},
        )
