"""Optional LLM Guard input/output scanning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GuardResult:
    allowed: bool
    sanitized_prompt: str
    sanitized_output: Optional[str] = None
    response_override: Optional[str] = None  # e.g. "Blocked"
    input_scores: Dict[str, Any] = field(default_factory=dict)
    output_scores: Dict[str, Any] = field(default_factory=dict)
    input_scan_time: float = 0.0
    output_scan_time: float = 0.0


class NoGuard:
    """Pass-through guard (no scanning)."""

    name = "none"

    def scan_input(self, prompt: str) -> GuardResult:
        return GuardResult(allowed=True, sanitized_prompt=prompt)

    def scan_output(self, prompt: str, output: str) -> GuardResult:
        return GuardResult(
            allowed=True, sanitized_prompt=prompt, sanitized_output=output
        )


class LLMGuard:
    """Wrap protectai/llm-guard scanners (lazy import)."""

    name = "llmguard"

    def __init__(
        self,
        *,
        toxicity_threshold: float = 0.5,
        injection_threshold: float = 0.5,
        ban_topics=None,
        ban_threshold: float = 0.5,
    ) -> None:
        try:
            from llm_guard import scan_output, scan_prompt
            from llm_guard.input_scanners import BanTopics, PromptInjection, Toxicity
        except ImportError as exc:
            raise ImportError(
                "llm-guard is not installed. Install with: "
                "pip install 'llm-tester[guard]' or pip install llm-guard"
            ) from exc

        self._scan_prompt = scan_prompt
        self._scan_output = scan_output
        topics = ban_topics or [
            "violence",
            "illegal activities",
            "hate speech",
            "gambling",
        ]
        self.input_scanners = [
            PromptInjection(threshold=injection_threshold),
            Toxicity(threshold=toxicity_threshold),
            BanTopics(topics=topics, threshold=ban_threshold),
        ]
        self.output_scanners = [
            Toxicity(threshold=toxicity_threshold),
        ]

    def scan_input(self, prompt: str) -> GuardResult:
        start = time.perf_counter()
        sanitized, results_valid, risk_scores = self._scan_prompt(
            self.input_scanners, prompt
        )
        elapsed = time.perf_counter() - start
        allowed = all(results_valid.values()) if results_valid else True
        return GuardResult(
            allowed=allowed,
            sanitized_prompt=sanitized if allowed else prompt,
            response_override=None if allowed else "Blocked",
            input_scores=dict(risk_scores) if risk_scores else {},
            input_scan_time=elapsed,
        )

    def scan_output(self, prompt: str, output: str) -> GuardResult:
        start = time.perf_counter()
        scanned, results_valid, risk_scores = self._scan_output(
            self.output_scanners, prompt, output
        )
        elapsed = time.perf_counter() - start
        allowed = all(results_valid.values()) if results_valid else True
        return GuardResult(
            allowed=allowed,
            sanitized_prompt=prompt,
            sanitized_output=scanned if allowed else output,
            response_override=None if allowed else "Blocked",
            output_scores=dict(risk_scores) if risk_scores else {},
            output_scan_time=elapsed,
        )


def build_guard(name: str, **kwargs) -> Any:
    key = (name or "none").lower().strip()
    if key in ("none", "off", "no"):
        return NoGuard()
    if key in ("llmguard", "llm-guard", "llm_guard", "guard"):
        return LLMGuard(**kwargs)
    raise ValueError(f"Unknown guard '{name}'. Choose: none, llmguard")
