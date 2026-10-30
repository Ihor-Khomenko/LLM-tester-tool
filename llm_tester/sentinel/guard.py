"""In-process MCP-Sentinel guard API."""

from __future__ import annotations

import re
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

        if not self.policy.enable_t0:
            elapsed = (time.perf_counter() - start) * 1000
            result = GuardDecision(
                decision=Decision.ALLOW,
                risk_score=0.0,
                stage="T0",
                checkpoint=Checkpoint.T0.value,
                reasons=["t0:disabled"],
                latency_ms=elapsed,
                meta={"t0_disabled": True},
            )
            self.logger.log(result)
            return result

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
        # Always record provenance for S2 even when T1 scanning is ablated.
        self.taint.mark_output(source, content or "")

        if not self.policy.enable_t1:
            elapsed = (time.perf_counter() - start) * 1000
            result = GuardDecision(
                decision=Decision.ALLOW,
                risk_score=0.0,
                stage="T1",
                checkpoint=Checkpoint.T1.value,
                reasons=["t1:disabled"],
                latency_ms=elapsed,
                sanitized_content=content,
                meta={"source": source, "t1_disabled": True},
            )
            self.logger.log(result)
            return result

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
            # Only label SANITIZE when the text actually changed (en+uk aware).
            sanitized = _sanitize_injection(content or "")
            if sanitized != (content or ""):
                decision = Decision.SANITIZE
                reasons.append("t1:sanitized")
            elif risk >= self.policy.tau_high:
                decision = Decision.BLOCK
                reasons.append("t1:block_output")
            else:
                # High S1 risk but sanitizer found nothing to strip — keep
                # ALLOW with elevated risk so cascade / T2 still see the signal.
                decision = Decision.ALLOW
                reasons.append("t1:sanitize_noop")
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
        spec = self.registry.lookup_spec(call.name)
        description = spec.description if spec else ""
        input_schema = spec.input_schema if spec else None

        if self.policy.enable_s1:
            stage_scores.append(
                self.rules.scan_call(
                    call, user_request=self.store.user_request
                )
            )
        if self.policy.enable_s2:
            stage_scores.append(
                self.taint.score_call(
                    call, description=description, input_schema=input_schema
                )
            )
        if self.policy.enable_s3 and self.intent is not None:
            stage_scores.append(
                self.intent.score_call(
                    self.store.user_request,
                    call,
                    description=description,
                    input_schema=input_schema,
                )
            )

        risk, reasons = aggregate_risk(stage_scores, self.policy)
        decision = decide_from_risk(
            risk, self.policy, stage_scores=stage_scores
        )
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
            reasons = list(reasons) + list(s4.reasons)
            s4_error = bool((s4.details or {}).get("decision_error")) or any(
                r in ("s4:error", "s4:parse_error") for r in (s4.reasons or [])
            )
            deferred = "s4:ambiguous_defer" in s4.reasons or bool(
                (s4.details or {}).get("deferred_to_prior")
            )
            if s4_error:
                # Provider/parse failure → fail closed (BLOCK).
                risk = max(float(s4.score), 0.95)
                decision = Decision.BLOCK
                final_stage = "S4"
                reasons = list(reasons) + ["s4:fail_closed"]
            elif deferred:
                # Keep pre-S4 cascade risk; do not treat ambiguous output as BLOCK.
                # S4 already ran — never re-escalate. Grey-zone / S3-uncertain → ALLOW.
                decision = decide_from_risk(
                    risk, self.policy, stage_scores=stage_scores
                )
                if decision == Decision.ESCALATE:
                    decision = Decision.ALLOW
                final_stage = "cascade"
            else:
                risk = s4.score
                decision = decide_from_risk(
                    risk,
                    self.policy,
                    has_s4_verdict=True,
                    s4_score=s4.score,
                    stage_scores=stage_scores,
                )
                final_stage = "S4"
        elif decision == Decision.ESCALATE:
            # S4 requested but unavailable → fail closed (block).
            decision = Decision.BLOCK
            reasons = list(reasons) + ["s4:unavailable_fail_closed"]
            final_stage = "cascade"

        # Dominant stage for reporting.
        if stage_scores:
            top = max(stage_scores, key=lambda s: s.score)
            if top.score >= risk * 0.9:
                final_stage = top.stage

        elapsed = (time.perf_counter() - start) * 1000
        llm_calls = 0
        decision_error = False
        for s in stage_scores:
            llm_calls += int((s.details or {}).get("llm_calls") or 0)
            if (s.details or {}).get("decision_error"):
                decision_error = True
        if any(
            str(r) in ("s4:error", "s4:parse_error", "s4:unavailable_fail_closed")
            or str(r).endswith(":error")
            for r in reasons
        ):
            decision_error = True
        if decision == Decision.BLOCK and "s4:unavailable_fail_closed" in reasons:
            decision_error = True
        result = GuardDecision(
            decision=decision,
            risk_score=float(min(1.0, risk)),
            stage=final_stage,
            checkpoint=Checkpoint.T2.value,
            reasons=reasons,
            latency_ms=elapsed,
            stage_scores=stage_scores,
            escalated=escalated,
            meta={
                "call": {"name": call.name, "arguments": call.arguments},
                "llm_calls": llm_calls,
                "decision_error": bool(decision_error),
                **(
                    {"error": True, "fail_closed": True}
                    if decision_error
                    else {}
                ),
            },
        )
        self.logger.log(result)
        return result


# Marker that starts an untrusted instruction block (en + uk, general classes —
# not scenario-literal strings). Used by ``_sanitize_injection``.
_INJECTION_BLOCK_START = re.compile(
    r"("
    # System / role-override headers
    r"\[\s*SYSTEM[^\]]*\]|"
    r"SYSTEM\s*:\s*CRITICAL\s*OVERRIDE|"
    r"<<\s*SYS\s*>>|"
    r"<\s*/?\s*IMPORTANT\s*>|"
    r"IMPORTANT!{1,3}|"
    r"#{2,}\s*ADMIN\s*MODE|"
    # Imperative ignore-previous (EN)
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|directives?)|"
    r"disregard\s+(all\s+)?(previous|prior|earlier)\s+\w+|"
    # Imperative ignore-previous (UK) — morphological, not a fixed sentence
    r"ігноруй(те)?\s+(вс[іi]\s+)?попередн\w*|"
    r"нехтуй(те)?\s+(вс[іi]\s+)?попередн\w*|"
    # Role-claim / compulsion markers (EN + UK)
    r"you\s+(are\s+no\s+longer|must\s+immediately|are\s+now\s+only)|"
    r"your\s+only\s+task\s+is|"
    r"ти\s+(більше\s+не\s+є|забов['’]?язаний|відтепер)|"
    r"твоє\s+єдине\s+завдання|"
    r"нове\s+правило\s*:"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def _sanitize_injection(text: str) -> str:
    """
    Neutralise untrusted instruction blocks in tool output (en + uk).

    **Principled behaviour:** when a general injection-block starter is found
    (system/role override header, ignore-previous imperative, or role-claim /
    compulsion marker), the **entire contiguous block** from that marker
    through the following numbered/bulleted instruction lines and imperative
    sentences is replaced with the placeholder
    ``[UNTRUSTED_INSTRUCTION_BLOCK_REMOVED]``. Preceding benign content
    (e.g. a legitimate report sentence before the override) is kept.

    This is best-effort provenance hygiene for downstream agents — **not** a
    security boundary. Returns the original string unchanged when no starter
    matched (callers must not label that as sanitized).
    """
    import re

    if not text:
        return text
    m = _INJECTION_BLOCK_START.search(text)
    if not m:
        # Soft single-line cleanups that are not full blocks.
        soft = [
            re.compile(
                r"do\s+not\s+(tell|inform|disclose)\s+(the\s+)?user[^\n]*",
                re.IGNORECASE,
            ),
            re.compile(
                r"не\s+(каж|повідомл|розголошуй)\w*[^\n]*",
                re.IGNORECASE,
            ),
            re.compile(
                r"прихован\w*\s+інструкц\w*[^\n]*",
                re.IGNORECASE,
            ),
            re.compile(
                r"(hidden|secret)\s+instruction[^\n]*",
                re.IGNORECASE,
            ),
        ]
        out = text
        for p in soft:
            out = p.sub(" ", out)
        cleaned = re.sub(r"[ \t]{2,}", " ", out)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    head = text[: m.start()].rstrip()
    # Consume the rest of the injection block: from the marker to EOF, or until
    # a blank line followed by clearly benign prose without imperatives —
    # default is to neutralize through end of string (safest for tool outputs).
    tail = text[m.start() :]
    # Keep a trailing non-imperative short footer if separated by 2+ blank lines
    # and lacking tool-call / override cues.
    parts = re.split(r"\n{3,}", tail, maxsplit=1)
    block = parts[0]
    footer = parts[1] if len(parts) > 1 else ""
    if footer and re.search(
        r"(tool|інструмент|must|забов|ignore|ігнору|system\s*:)",
        footer,
        re.I,
    ):
        block = tail
        footer = ""
    placeholder = "[UNTRUSTED_INSTRUCTION_BLOCK_REMOVED]"
    pieces = [p for p in (head, placeholder, footer.strip()) if p]
    return "\n\n".join(pieces).strip()


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
