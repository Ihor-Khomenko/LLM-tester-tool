"""
Optional LLM provider adapter interface for MCP-Sentinel.

Sentinel itself has **no** dependency on ``llm_tester.providers``. Any object
that implements :class:`LLMChat` can be injected for S3 (LLM intent) / S4
(judge). The host package (or a future standalone ``mcp-sentinel``) supplies
the concrete client — e.g. an adapter around OpenAI-compatible APIs
(Ollama, OpenRouter ``:free``, xAI, Groq, OpenAI).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMChat(Protocol):
    """Minimal chat interface expected by S3/S4 stages."""

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
    ) -> Any:
        """
        Return either a string or an object with a ``.content`` attribute
        (e.g. ``llm_tester.providers.ChatResult``).
        """
        ...


def extract_chat_text(result: Any) -> str:
    """Normalise provider responses to plain text."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if content is not None:
        return str(content)
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)
