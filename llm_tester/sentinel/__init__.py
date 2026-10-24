"""
MCP-Sentinel — cascaded provenance-aware runtime guard for agentic MCP hosts.

Self-contained subpackage: no imports from the rest of ``llm_tester`` except an
optional duck-typed :class:`~llm_tester.sentinel.provider.LLMChat` injected by
the host. Designed for later extraction into a standalone ``mcp-sentinel`` repo.

Evaluation CLIs (``guard-eval``, ``agent-bench``) live in ``llm_tester.agent``.
"""

from llm_tester.sentinel.guard import SentinelGuard, build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy, load_policy
from llm_tester.sentinel.provider import LLMChat, extract_chat_text
from llm_tester.sentinel.types import Decision, GuardDecision, ToolCall, ToolSpec

__all__ = [
    "SentinelGuard",
    "build_sentinel",
    "SentinelPolicy",
    "load_policy",
    "LLMChat",
    "extract_chat_text",
    "Decision",
    "GuardDecision",
    "ToolCall",
    "ToolSpec",
]
