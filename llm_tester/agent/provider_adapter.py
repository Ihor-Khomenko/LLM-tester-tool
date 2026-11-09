"""
Bridge ``llm_tester.providers.LLMProvider`` → sentinel :class:`LLMChat`.

Sentinel never imports this module; the CLI / agent-bench does.
Any OpenAI-compatible provider registered in ``llm_tester.config.PROVIDERS``
works: ``ollama``, ``openai``, ``xai``, ``openrouter`` (incl. ``:free`` models),
``groq``.
"""

from __future__ import annotations

from typing import Any, Optional

from llm_tester.config import DEFAULT_SEED, DEFAULT_TEMPERATURE, PROVIDERS, RunParams
from llm_tester.providers import LLMProvider
from llm_tester.sentinel.provider import LLMChat  # noqa: F401 — re-export for typing


def build_llm_chat(
    provider: str,
    model: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = DEFAULT_SEED,
    max_retries: int = 5,
    max_tokens: Optional[int] = None,
    num_thread: Optional[int] = None,
) -> LLMProvider:
    """
    Construct an OpenAI-compatible chat client usable as sentinel ``llm_provider``.

    ``LLMProvider.chat(prompt, system=...)`` already matches the ``LLMChat``
    protocol, so the instance can be passed directly to
    :func:`llm_tester.sentinel.build_sentinel`.

    Any OpenAI-compatible endpoint works: pass ``provider='ollama'`` (or
    ``openai``) with ``base_url='http://localhost:11434/v1'`` and an optional
    ``api_key`` (Ollama accepts any dummy key; default ``ollama``).

    For Ollama, ``num_thread`` (and seed/temperature) are also sent under
    ``extra_body.options`` so native sampling knobs are explicit.
    """
    name = provider.lower().strip()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {', '.join(sorted(PROVIDERS))}"
        )
    params = RunParams(
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
        num_thread=num_thread,
    )
    return LLMProvider(
        name,
        model,
        base_url=base_url,
        api_key=api_key,
        params=params,
        max_retries=max_retries,
    )


def provider_choices() -> list:
    return sorted(PROVIDERS.keys())
