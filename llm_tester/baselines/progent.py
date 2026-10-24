"""Progent-style static privilege policy baseline.

Derives a coarse allowlist of tool *capabilities* from the user request and
blocks side-effect / privileged tools that fall outside that policy. Offline
and deterministic (no LLM).
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, FrozenSet, List, Optional, Set, Union

from llm_tester.baselines.protocol import BaseGuard, allow_decision, as_tool_call
from llm_tester.sentinel.side_effects import is_side_effect_tool
from llm_tester.sentinel.types import Checkpoint, Decision, GuardDecision, ToolCall

# Capability tags inferred from the user utterance.
_CAP_PATTERNS = (
    ("read", re.compile(r"\b(read|summaris|summariz|show|list|get|fetch|open|view)\b", re.I)),
    ("write", re.compile(r"\b(write|create|save|append|update|edit|organise|organize)\b", re.I)),
    ("email", re.compile(r"\b(email|mail|send\s+message|inbox)\b", re.I)),
    ("money", re.compile(r"\b(transfer|pay|wire|iban|bank|funds?)\b", re.I)),
    ("shell", re.compile(r"\b(shell|terminal|command|bash|exec)\b", re.I)),
    ("http", re.compile(r"\b(http|url|download|upload|api)\b", re.I)),
)

_TOOL_CAPS: Dict[str, FrozenSet[str]] = {
    "notes.read": frozenset({"read"}),
    "notes.write": frozenset({"write"}),
    "read_text_file": frozenset({"read"}),
    "write_file": frozenset({"write"}),
    "list_files": frozenset({"read"}),
    "email.send": frozenset({"email"}),
    "email.read": frozenset({"email", "read"}),
    "bank.transfer": frozenset({"money"}),
    "shell.exec": frozenset({"shell"}),
    "http.fetch": frozenset({"http", "read"}),
}


def infer_capabilities(user_request: str) -> Set[str]:
    caps: Set[str] = set()
    text = user_request or ""
    for name, pat in _CAP_PATTERNS:
        if pat.search(text):
            caps.add(name)
    # Default: read-only if nothing matched.
    if not caps:
        caps.add("read")
    return caps


def tool_capabilities(tool_name: str) -> Set[str]:
    key = (tool_name or "").lower().strip()
    if key in _TOOL_CAPS:
        return set(_TOOL_CAPS[key])
    # Heuristic from name segments.
    parts = set(re.split(r"[._\-\s/]+", key))
    caps: Set[str] = set()
    if parts & {"read", "list", "get", "fetch", "open", "cat", "load"}:
        caps.add("read")
    if parts & {"write", "create", "save", "update", "append", "put"}:
        caps.add("write")
    if parts & {"email", "mail", "send"} and "money" not in parts:
        caps.add("email")
    if parts & {"transfer", "pay", "wire", "bank"}:
        caps.add("money")
    if parts & {"shell", "exec", "bash", "command"}:
        caps.add("shell")
    if parts & {"http", "url", "download", "upload"}:
        caps.add("http")
    if not caps and is_side_effect_tool(tool_name):
        caps.add("write")
    if not caps:
        caps.add("read")
    return caps


class ProgentGuard(BaseGuard):
    """Static privilege rules: side-effect tools need matching user capabilities."""

    name = "progent"

    def __init__(self, *, strict: bool = True) -> None:
        super().__init__()
        self.strict = strict
        self._caps: Set[str] = {"read"}

    def set_user_task(self, text: str) -> None:
        super().set_user_task(text)
        self._caps = infer_capabilities(text)

    def on_tool_output(
        self, source: str, content: str, *, sanitize: bool = True
    ) -> GuardDecision:
        start = time.perf_counter()
        self._note_output(source, content or "")
        # Privilege policy does not inspect content — always allow T1.
        return allow_decision(
            checkpoint=Checkpoint.T1.value,
            stage=self.name,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=["progent:t1_passthrough"],
            meta={"capabilities": sorted(self._caps), "llm_calls": 0},
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
        needed = tool_capabilities(tc.name)
        allowed = self._caps
        missing = needed - allowed
        side = is_side_effect_tool(tc.name, arguments=tc.arguments)
        risk = 0.1
        decision = Decision.ALLOW
        reasons = [f"progent:caps={sorted(allowed)}", f"progent:needed={sorted(needed)}"]
        if missing and side:
            risk = 0.9
            decision = Decision.BLOCK
            reasons.append(f"progent:privilege_violation:{sorted(missing)}")
        elif missing and self.strict:
            risk = 0.75
            decision = Decision.BLOCK
            reasons.append(f"progent:capability_missing:{sorted(missing)}")
        elif missing:
            risk = 0.45
            reasons.append(f"progent:soft_missing:{sorted(missing)}")
        return allow_decision(
            checkpoint=Checkpoint.T2.value,
            stage=self.name,
            risk=risk,
            latency_ms=(time.perf_counter() - start) * 1000,
            reasons=reasons,
            decision=decision,
            meta={
                "capabilities": sorted(allowed),
                "needed": sorted(needed),
                "missing": sorted(missing),
                "llm_calls": 0,
            },
        )
