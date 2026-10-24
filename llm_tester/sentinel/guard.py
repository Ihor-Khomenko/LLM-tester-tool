"""In-process MCP-Sentinel guard API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from llm_tester.sentinel.cascade import aggregate_risk, decide_from_risk
from llm_tester.sentinel.logging_util import DecisionLogger
from llm_tester.sentinel.policy import SentinelPolicy, load_policy
from llm_tester.sentinel.registry import ToolRegistry
from llm_tester.sentinel.stages.intent import IntentStage
from llm_tester.sentinel.stages.judge import JudgeStage
from llm_tester.sentinel.stages.rules import RulesStage
from llm_tester.sentinel.stages.taint import TaintStage, TaintStore
from llm_tester.sentinel.types import (
    Checkpoint,
    Decision,
    GuardDecision,
    StageScore,
    ToolCall,
    ToolSpec,
)


class SentinelGuard:
    """
    Cascaded provenance-aware runtime guard.

    Typical flow::

        guard = SentinelGuard()
        guard.set_user_task("Pay Alice 10 EUR")
        guard.check_tool_registration([ToolSpec(...)])
        guard.on_tool_output("email.read", raw_email)
        decision = guard.check_tool_call(ToolCall("transfer", {...}))
    """

    def __init__(
        self,
        policy: Optional[SentinelPolicy] = None,
        *,
        policy_path: Optional[Union[str, Path]] = None,
        llm_provider: Any = None,
        judge_provider: Any = None,
        log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if policy is None:
            policy = load_policy(Path(policy_path) if policy_path else None)
        self.policy = policy
        self.registry = ToolRegistry()
        self.store = TaintStore(n=policy.taint_ngram_n)
        self.rules = RulesStage(policy)
        self.taint = TaintStage(policy, self.store)
        # Prefer shared llm_provider; judge_provider kept as alias for S4-only inject.
        chat = llm_provider if llm_provider is not None else judge_provider
        self.intent = (
            IntentStage(policy, provider=chat) if policy.enable_s3 else None
        )
        self.judge = (
            JudgeStage(policy, provider=chat) if policy.enable_s4 else None
        )
        path = log_path or policy.log_path
        self.logger = DecisionLogger(Path(path) if path else None)

    def set_user_task(self, text: str) -> None:
        """SDK hook: declare the trusted user request for provenance."""
        self.store.set_user_task(text or "")

    def reset_session(self, *, keep_registry: bool = True) -> None:
        self.store.clear_outputs()
        if not keep_registry:
            self.registry = ToolRegistry()

    # --- T0 -----------------------------------------------------------------

    def check_tool_registration(
        self,
        tools: Sequence[Union[ToolSpec, Dict[str, Any]]],
    ) -> GuardDecision:
        start = time.perf_counter()
        reasons: List[str] = []
        stage_scores: List[StageScore] = []
        peak = 0.0
        details: Dict[str, Any] = {"tools": []}

        for raw in tools:
            tool = raw if isinstance(raw, ToolSpec) else ToolSpec(
                name=str(raw.get("name", "")),
                description=str(raw.get("description", "")),
                input_schema=raw.get("inputSchema") or raw.get("input_schema"),
            )
            t0_risk, t0_reasons, t0_details = self.registry.check(
                tool,
                semantic_diff_threshold=self.policy.t0_semantic_diff_threshold,
                block_on_hash_mismatch=self.policy.t0_block_on_hash_mismatch,
            )
            s1 = (
                self.rules.scan_text(tool.description)
                if self.policy.enable_s1
                else StageScore(stage="S1", score=0.0)
            )
            tool_risk = max(t0_risk, s1.score)
            peak = max(peak, tool_risk)
            reasons.extend(t0_reasons)
            reasons.extend(s1.reasons)
            stage_scores.append(s1)
            details["tools"].append(
                {"name": tool.name, "t0": t0_details, "s1": s1.to_dict()}
            )

        risk = float(min(1.0, peak))
        if risk >= self.policy.tau_high:
            decision = Decision.BLOCK
            stage = "T0"
        elif risk <= self.policy.tau_low:
            decision = Decision.ALLOW
            stage = "T0"
        else:
            decision = Decision.ASK_USER if self.policy.ask_user_on_block else Decision.BLOCK
            stage = "T0"

        elapsed = (time.perf_counter() - start) * 1000
        result = GuardDecision(
            decision=decision,
            risk_score=risk,
            stage=stage,
            checkpoint=Checkpoint.T0.value,
            reasons=reasons,
            latency_ms=elapsed,
            stage_scores=stage_scores,
            meta=details,
        )
        self.logger.log(result)
        return result

    # --- T1 -----------------------------------------------------------------

    def on_tool_output(
        self,
        source: str,
        content: str,
        *,
        sanitize: bool = True,
    ) -> GuardDecision:
        start = time.perf_counter()
        self.taint.mark_output(source, content or "")

        s1 = (
            self.rules.scan_text(content or "")
            if self.policy.enable_s1
            else StageScore(stage="S1", score=0.0)
        )
        risk = s1.score
        reasons = list(s1.reasons)
        sanitized = content
        decision = Decision.ALLOW

        if risk >= self.policy.s1_block_threshold and sanitize:
            # Mark but do not drop entirely — provenance still needed for S2.
            # Optionally strip high-signal injection phrases for downstream agent.
            sanitized = _sanitize_injection(content or "")
            decision = Decision.SANITIZE
            reasons.append("t1:sanitized")
        elif risk >= self.policy.tau_high:
            decision = Decision.BLOCK
            reasons.append("t1:block_output")

        elapsed = (time.perf_counter() - start) * 1000
        result = GuardDecision(
            decision=decision,
            risk_score=float(min(1.0, risk)),
            stage="S1" if s1.reasons else "T1",
            checkpoint=Checkpoint.T1.value,
            reasons=reasons,
            latency_ms=elapsed,
            stage_scores=[s1],
            sanitized_content=sanitized,
            meta={"source": source},
        )
        self.logger.log(result)
        return result

    # --- T2 -----------------------------------------------------------------

    def check_tool_call(
        self,
        call: Union[ToolCall, Dict[str, Any]],
        *,
        user_request: Optional[str] = None,
    ) -> GuardDecision:
        start = time.perf_counter()
        if isinstance(call, dict):
            call = ToolCall(
                name=str(call.get("name", "")),
                arguments=dict(call.get("arguments") or call.get("args") or {}),
            )
        if user_request is not None:
            self.set_user_task(user_request)

        stage_scores: List[StageScore] = []

        if self.policy.enable_s1:
            stage_scores.append(self.rules.scan_call(call))
        if self.policy.enable_s2:
            stage_scores.append(self.taint.score_call(call))
        if self.policy.enable_s3 and self.intent is not None:
            stage_scores.append(
                self.intent.score_call(self.store.user_request, call)
            )

        risk, reasons = aggregate_risk(stage_scores, self.policy)
        decision = decide_from_risk(risk, self.policy)
        escalated = False
        final_stage = "cascade"

        if decision == Decision.ESCALATE and self.judge is not None and self.judge.available:
            escalated = True
            snippets = [s.content for s in self.store.sources[-5:]]
            s4 = self.judge.judge(
                self.store.user_request,
                call,
                snippets,
                risk,
                reasons,
            )
            stage_scores.append(s4)
            risk = s4.score
            reasons = list(reasons) + list(s4.reasons)
            decision = decide_from_risk(
                risk,
                self.policy,
                has_s4_verdict=True,
                s4_score=s4.score,
            )
            final_stage = "S4"
        elif decision == Decision.ESCALATE:
            # No judge available: conservative mid-threshold policy already applied
            # by decide_from_risk when enable_s4 is false; if enable_s4 but unavailable:
            mid = (self.policy.tau_low + self.policy.tau_high) / 2.0
            decision = Decision.BLOCK if risk >= mid else Decision.ALLOW
            reasons = list(reasons) + ["s4:skipped"]
            final_stage = "cascade"

        # Dominant stage for reporting.
        if stage_scores:
            top = max(stage_scores, key=lambda s: s.score)
            if top.score >= risk * 0.9:
                final_stage = top.stage

        elapsed = (time.perf_counter() - start) * 1000
        result = GuardDecision(
            decision=decision,
            risk_score=float(min(1.0, risk)),
            stage=final_stage,
            checkpoint=Checkpoint.T2.value,
            reasons=reasons,
            latency_ms=elapsed,
            stage_scores=stage_scores,
            escalated=escalated,
            meta={"call": {"name": call.name, "arguments": call.arguments}},
        )
        self.logger.log(result)
        return result


def _sanitize_injection(text: str) -> str:
    """Best-effort strip of common injection wrappers (not a security boundary)."""
    import re

    patterns = [
        re.compile(
            r"IMPORTANT!!!\s*Ignore\s+all\s+previous[^\n]*\n?",
            re.IGNORECASE,
        ),
        re.compile(r"<\s*/?\s*IMPORTANT\s*>", re.IGNORECASE),
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)[^\n]*",
            re.IGNORECASE,
        ),
    ]
    out = text
    for p in patterns:
        out = p.sub(" ", out)
    return out.strip()


def build_sentinel(
    *,
    policy_path: Optional[Union[str, Path]] = None,
    policy: Optional[SentinelPolicy] = None,
    enable_s3: Optional[bool] = None,
    enable_s4: Optional[bool] = None,
    llm_provider: Any = None,
    judge_provider: Any = None,
    log_path: Optional[Union[str, Path]] = None,
    ablation: Optional[Dict[str, bool]] = None,
) -> SentinelGuard:
    """Factory used by CLI and library callers.

    ``llm_provider`` must satisfy :class:`~llm_tester.sentinel.provider.LLMChat`
    (duck-typed). The host wires OpenAI-compatible clients (Ollama, OpenRouter,
    xAI, Groq, OpenAI); Sentinel never imports those SDKs itself.
    """
    pol = policy or load_policy(Path(policy_path) if policy_path else None)
    if enable_s3 is not None:
        pol.enable_s3 = enable_s3
    if enable_s4 is not None:
        pol.enable_s4 = enable_s4
    if ablation:
        pol = pol.with_ablation(**ablation)
    return SentinelGuard(
        pol,
        llm_provider=llm_provider,
        judge_provider=judge_provider,
        log_path=log_path,
    )
