"""S1 — fast rules & signatures (< 1 ms target)."""

from __future__ import annotations

import base64
import re
import time
from typing import List, Optional, Sequence, Tuple

from llm_tester.sentinel.policy import SentinelPolicy
from llm_tester.sentinel.types import StageScore, ToolCall

# Injection / exfiltration signatures (case-insensitive where noted).
INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern, float], ...] = (
    (
        "ignore_previous",
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
            re.IGNORECASE,
        ),
        0.95,
    ),
    (
        "important_tag",
        re.compile(r"<\s*/?\s*IMPORTANT\s*>", re.IGNORECASE),
        0.9,
    ),
    (
        "system_override",
        re.compile(
            r"(system\s*:\s*|\[SYSTEM\]|<<\s*SYS\s*>>)",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "hacking_prompt",
        re.compile(
            r"IMPORTANT!!!\s*Ignore\s+all\s+previous",
            re.IGNORECASE,
        ),
        0.98,
    ),
    (
        "do_not_tell_user",
        re.compile(
            r"do\s+not\s+(tell|inform|disclose)\s+(the\s+)?user",
            re.IGNORECASE,
        ),
        0.8,
    ),
    (
        "exfil_markdown_image",
        re.compile(
            r"!\[[^\]]*\]\(\s*https?://[^\)]*(?:key|token|secret|password)=",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "exfil_url_params",
        re.compile(
            r"https?://[^\s\"']+[?&](api_?key|token|password|secret)=",
            re.IGNORECASE,
        ),
        0.85,
    ),
    (
        "hidden_instruction",
        re.compile(
            r"(hidden|secret)\s+instruction",
            re.IGNORECASE,
        ),
        0.75,
    ),
)

# Zero-width / invisible Unicode that can hide instructions.
INVISIBLE_UNICODE = re.compile(
    r"[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u2066-\u2069]"
)

# Long base64-looking blobs (potential payload smuggling).
BASE64_BLOB = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/])")


def _looks_like_base64_payload(blob: str) -> bool:
    try:
        raw = base64.b64decode(blob, validate=False)
    except Exception:
        return False
    if len(raw) < 16:
        return False
    # Decoded text containing injection keywords → suspicious.
    try:
        text = raw.decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    needles = ("ignore", "system", "instruction", "password", "http://", "https://")
    return any(n in text for n in needles)


class RulesStage:
    """Signature / regex scanner used at T0 (descriptions), T1 (outputs), T2 (args)."""

    name = "S1"

    def __init__(self, policy: Optional[SentinelPolicy] = None) -> None:
        self.policy = policy or SentinelPolicy()

    def scan_text(self, text: str) -> StageScore:
        start = time.perf_counter()
        reasons: List[str] = []
        score = 0.0
        details = {"matches": []}

        if not text:
            return StageScore(
                stage=self.name,
                score=0.0,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        for name, pattern, weight in INJECTION_PATTERNS:
            if pattern.search(text):
                reasons.append(f"s1:{name}")
                details["matches"].append(name)
                score = max(score, weight)

        invis = INVISIBLE_UNICODE.findall(text)
        if invis:
            reasons.append("s1:invisible_unicode")
            details["matches"].append("invisible_unicode")
            details["invisible_count"] = len(invis)
            score = max(score, 0.7)

        for blob in BASE64_BLOB.findall(text):
            if _looks_like_base64_payload(blob):
                reasons.append("s1:base64_payload")
                details["matches"].append("base64_payload")
                score = max(score, 0.75)
                break

        elapsed = (time.perf_counter() - start) * 1000
        return StageScore(
            stage=self.name,
            score=min(1.0, score),
            reasons=reasons,
            details=details,
            latency_ms=elapsed,
        )

    def scan_call(self, call: ToolCall) -> StageScore:
        """Scan tool name + serialized arguments; apply destructive-tool policy."""
        start = time.perf_counter()
        parts = [call.name] + [f"{k}={v}" for k, v in call.arguments.items()]
        base = self.scan_text("\n".join(parts))
        reasons = list(base.reasons)
        details = dict(base.details)
        score = base.score

        if self.policy.block_destructive_tools:
            name_l = call.name.lower()
            for sub in self.policy.destructive_tool_substrings:
                if sub.lower() in name_l:
                    # Soft signal — destructive alone is not an attack, but
                    # elevates risk when combined with taint (handled in cascade).
                    reasons.append(f"s1:destructive_tool:{sub}")
                    details["destructive"] = True
                    score = max(score, 0.35)
                    break

        elapsed = (time.perf_counter() - start) * 1000
        return StageScore(
            stage=self.name,
            score=min(1.0, score),
            reasons=reasons,
            details=details,
            latency_ms=elapsed + base.latency_ms,
        )

    def scan_many(self, texts: Sequence[str]) -> StageScore:
        start = time.perf_counter()
        best = StageScore(stage=self.name, score=0.0)
        for t in texts:
            s = self.scan_text(t)
            if s.score >= best.score:
                best = s
        best.latency_ms = (time.perf_counter() - start) * 1000
        return best
