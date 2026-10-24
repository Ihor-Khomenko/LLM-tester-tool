"""Spotlighting baseline (Hines et al. 2024) — datamark / encode tool outputs.

Untrusted tool outputs are wrapped with datamarkers so a downstream LLM can
treat them as data. Detection metrics use lightweight injection signatures on
the *unmarked* text; the marked form is returned as ``sanitized_content``.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Union

from llm_tester.baselines.protocol import BaseGuard, allow_decision, as_tool_call
from llm_tester.sentinel.types import Checkpoint, Decision, GuardDecision, ToolCall

# Simple injection cues (offline, no LLM).
_INJECTION = re.compile(
    r"("
    r"ignore\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)"
    r"|<\s*/?\s*IMPORTANT\s*>"
    r"|do\s+not\s+(tell|inform)\s+(the\s+)?user"
    r"|system\s*:\s*"
    r")",
    re.IGNORECASE,
)

_DATAMARK = "\u241f"  # unit separator glyph — rare in natural text


def datamark_text(text: str, *, marker: str = _DATAMARK) -> str:
    """Interleave a datamarker between tokens (Hines-style datamarking)."""
    if not text:
        return text
    # Token-level marking keeps readability while breaking instruction continuity.
    parts = re.split(r"(\s+)", text)
    out: List[str] = []
    for p in parts:
        if not p or p.isspace():
            out.append(p)
            continue
        out.append(marker.join(list(p)))
    return "".join(out)


def encode_text(text: str) -> str:
    """Base64-style encoding wrapper (Spotlighting encoding variant)."""
    import base64

    raw = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
    return f"<<ENCODED_TOOL_OUTPUT>>{raw}<</ENCODED_TOOL_OUTPUT>>"


def injection_risk(text: str) -> float:
    if not text:
        return 0.0
    if _INJECTION.search(text):
        return 0.92
    # Soft cues
    lower = text.lower()
    soft = 0.0
    for needle, w in (
        ("exfiltrat", 0.55),
        ("api_key", 0.5),
        ("password", 0.45),
        ("transfer funds", 0.6),
    ):
        if needle in lower:
            soft = max(soft, w)
    return soft


class SpotlightingGuard(BaseGuard):
    """Datamark (default) or encode untrusted outputs; flag injection cues."""

    name = "spotlighting"

    def __init__(self, *, mode: str = "datamark") -> None:
        super().__init__()
        mode_l = (mode or "datamark").lower().strip()
        if mode_l not in ("datamark", "encode"):
            raise ValueError("spotlighting mode must be datamark|encode")
        self.mode = mode_l

    def _transform(self, content: str) -> str:
        if self.mode == "encode":
            return encode_text(content)
        return datamark_text(content)

    def on_tool_output(
        self, source: str, content: str, *, sanitize: bool = True
    ) -> GuardDecision:
        start = time.perf_counter()
        raw = content or ""
        self._note_output(source, raw)
        marked = self._transform(raw) if sanitize else raw
        risk = injection_risk(raw)
        decision = Decision.ALLOW
        reasons: List[str] = [f"spotlighting:{self.mode}"]
        if risk >= 0.7:
            decision = Decision.SANITIZE
            reasons.append("spotlighting:injection_cue")
        return allow_decision(
            checkpoint=Checkpoint.T1.value,
            stage=self.name,
            risk=risk,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=reasons,
            decision=decision,
            sanitized_content=marked,
            meta={"mode": self.mode, "llm_calls": 0},
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
        # If args echo unmarked injection from prior outputs → block.
        blob = " ".join(str(v) for v in (tc.arguments or {}).values())
        risk = injection_risk(blob)
        # Also elevate when prior T1 saw high-risk content and call is side-effectish.
        prior = max((injection_risk(o["content"]) for o in self.outputs), default=0.0)
        name_l = tc.name.lower()
        side = any(
            v in name_l
            for v in ("send", "transfer", "write", "exec", "shell", "http", "delete")
        )
        if prior >= 0.7 and side:
            risk = max(risk, 0.85)
        decision = Decision.BLOCK if risk >= 0.7 else Decision.ALLOW
        reasons = []
        if decision == Decision.BLOCK:
            reasons.append("spotlighting:tainted_side_effect")
        return allow_decision(
            checkpoint=Checkpoint.T2.value,
            stage=self.name,
            risk=risk,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=reasons or ["spotlighting:allow"],
            decision=decision,
            meta={"llm_calls": 0},
        )
