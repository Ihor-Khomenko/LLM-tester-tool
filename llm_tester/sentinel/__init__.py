"""
Оберіг (Oberih) — cascaded provenance-aware runtime guard for agentic MCP hosts.

Formerly **MCP-Sentinel**. Canonical machine id: ``oberih``. Package path
``llm_tester.sentinel`` and legacy imports (``SentinelGuard``,
``build_sentinel``, …) remain stable aliases.

Self-contained subpackage: no imports from the rest of ``llm_tester`` except an
optional duck-typed :class:`~llm_tester.sentinel.provider.LLMChat` injected by
the host. Prefer ``import llm_tester.oberih`` for new code; this package stays
for compatibility and possible later extraction.

Evaluation CLIs (``guard-eval``, ``agent-bench``) live in ``llm_tester.agent``.
"""

from llm_tester.sentinel.branding import (
    DISPLAY_NAME,
    DISPLAY_NAME_EN,
    DISPLAY_NAME_UK,
    LEGACY_DISPLAY_NAME,
    LEGACY_MACHINE_ID,
    MACHINE_ID,
    product_blurb,
)
from llm_tester.sentinel.guard import (
    OberihGuard,
    SentinelGuard,
    build_oberih,
    build_sentinel,
)
from llm_tester.sentinel.policy import OberihPolicy, SentinelPolicy, load_policy
from llm_tester.sentinel.provider import LLMChat, extract_chat_text
from llm_tester.sentinel.types import Decision, GuardDecision, ToolCall, ToolSpec

__all__ = [
    "DISPLAY_NAME",
    "DISPLAY_NAME_EN",
    "DISPLAY_NAME_UK",
    "LEGACY_DISPLAY_NAME",
    "LEGACY_MACHINE_ID",
    "MACHINE_ID",
    "product_blurb",
    "OberihGuard",
    "SentinelGuard",
    "build_oberih",
    "build_sentinel",
    "OberihPolicy",
    "SentinelPolicy",
    "load_policy",
    "LLMChat",
    "extract_chat_text",
    "Decision",
    "GuardDecision",
    "ToolCall",
    "ToolSpec",
]
