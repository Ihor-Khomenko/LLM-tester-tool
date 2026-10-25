"""OpenAI-compatible multi-provider client layer."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import APIStatusError, OpenAI, RateLimitError

from llm_tester.config import RunParams, resolve_provider_config


@dataclass
class ChatResult:
    content: str
    latency_s: float
    raw: Optional[Dict[str, Any]] = None
    attempts: int = 1


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
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 60.0,
    ) -> None:
        cfg = resolve_provider_config(provider, base_url=base_url, api_key=api_key)
        self.provider = cfg["name"]
        self.base_url = cfg["base_url"]
        self.model = model or cfg["default_model"]
        if not self.model:
            raise ValueError(f"Model is required for provider '{self.provider}'.")
        self.params = params or RunParams()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        # Last chat() attempt count (incl. 429 retries) for --max-requests.
        self.last_attempts: int = 0
        # Disable the SDK's own retries so we control 429 backoff centrally.
        self._client = OpenAI(
            api_key=cfg["api_key"],
            base_url=self.base_url,
            max_retries=0,
        )

    def chat(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        stop: Optional[List[str]] = None,
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
        if stop:
            kwargs["stop"] = list(stop)

        attempt = 0
        start = time.perf_counter()
        while True:
            attempt += 1
            self.last_attempts = attempt
            try:
                response = self._client.chat.completions.create(**kwargs)
                latency = time.perf_counter() - start
                content = ""
                if response.choices:
                    content = response.choices[0].message.content or ""
                return ChatResult(
                    content=content,
                    latency_s=latency,
                    raw=response.model_dump(),
                    attempts=attempt,
                )
            except (RateLimitError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if status is None and hasattr(exc, "response"):
                    status = getattr(exc.response, "status_code", None)
                is_rate_limited = isinstance(exc, RateLimitError) or status == 429
                if not is_rate_limited or attempt > self.max_retries:
                    # Exhausted / non-429: still record attempts for budgeting.
                    self.last_attempts = attempt
                    raise
                delay = self._backoff_seconds(attempt, exc)
                time.sleep(delay)

    def _backoff_seconds(self, attempt: int, exc: BaseException) -> float:
        """Exponential backoff with jitter; honour Retry-After when present."""
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            headers = getattr(response, "headers", None) or {}
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw is not None:
                try:
                    retry_after = float(raw)
                except (TypeError, ValueError):
                    retry_after = None
        if retry_after is not None:
            return min(self.backoff_max, max(0.0, retry_after))
        expo = self.backoff_base * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.backoff_base)
        return min(self.backoff_max, expo + jitter)

    def info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "params": self.params.as_dict(),
        }
