"""OpenAI-compatible multi-provider client layer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from llm_tester.config import RunParams, resolve_provider_config


@dataclass
class ChatResult:
    content: str
    latency_s: float
    raw: Optional[Dict[str, Any]] = None


class LLMProvider:
    """Thin wrapper around the OpenAI SDK pointed at a configurable base_url."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        params: Optional[RunParams] = None,
    ) -> None:
        cfg = resolve_provider_config(provider, base_url=base_url, api_key=api_key)
        self.provider = cfg["name"]
        self.base_url = cfg["base_url"]
        self.model = model or cfg["default_model"]
        if not self.model:
            raise ValueError(f"Model is required for provider '{self.provider}'.")
        self.params = params or RunParams()
        self._client = OpenAI(api_key=cfg["api_key"], base_url=self.base_url)

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> ChatResult:
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.params.temperature,
        }
        if self.params.seed is not None:
            kwargs["seed"] = self.params.seed
        if self.params.max_tokens is not None:
            kwargs["max_tokens"] = self.params.max_tokens

        start = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - start
        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""
        return ChatResult(content=content, latency_s=latency, raw=response.model_dump())

    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "params": self.params.as_dict(),
        }
