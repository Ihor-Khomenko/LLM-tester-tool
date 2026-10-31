"""OpenAI-compatible multi-provider client layer."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import APIStatusError, OpenAI, RateLimitError

from llm_tester.config import (
    DEFAULT_OLLAMA_NUM_THREAD,
    RunParams,
    resolve_provider_config,
)


@dataclass
class ChatResult:
    content: str
    latency_s: float
    raw: Optional[Dict[str, Any]] = None
    attempts: int = 1
    n_429: int = 0
    # Echo of sampling knobs actually sent (for S4 / Ollama determinism checks).
    request_options: Optional[Dict[str, Any]] = None


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
        # 429 responses observed during the last chat() (retries + final fail).
        self.last_n_429: int = 0
        # Last request kwargs (incl. Ollama options) for tests / debugging.
        self.last_request_kwargs: Dict[str, Any] = {}
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
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        num_thread: Optional[int] = None,
    ) -> ChatResult:
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        temp = (
            float(temperature)
            if temperature is not None
            else float(self.params.temperature)
        )
        seed_val = seed if seed is not None else self.params.seed
        thread_val = (
            num_thread if num_thread is not None else self.params.num_thread
        )

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
        }
        if seed_val is not None:
            kwargs["seed"] = int(seed_val)
        if self.params.max_tokens is not None:
            kwargs["max_tokens"] = self.params.max_tokens
        if stop:
            kwargs["stop"] = list(stop)

        # Ollama: also push options.* so native sampling knobs are explicit
        # (seed / temperature / num_thread). OpenAI-compat maps top-level
        # temperature/seed, but num_thread only exists under options.
        request_options: Dict[str, Any] = {
            "temperature": temp,
        }
        if seed_val is not None:
            request_options["seed"] = int(seed_val)
        if self.provider == "ollama":
            if thread_val is None and seed_val is not None:
                thread_val = DEFAULT_OLLAMA_NUM_THREAD
            if thread_val is not None:
                request_options["num_thread"] = int(thread_val)
            kwargs["extra_body"] = {"options": dict(request_options)}

        self.last_request_kwargs = dict(kwargs)

        attempt = 0
        n_429 = 0
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
                self.last_n_429 = n_429
                return ChatResult(
                    content=content,
                    latency_s=latency,
                    raw=response.model_dump(),
                    attempts=attempt,
                    n_429=n_429,
                    request_options=dict(request_options),
                )
            except (RateLimitError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                if status is None and hasattr(exc, "response"):
                    status = getattr(exc.response, "status_code", None)
                is_rate_limited = isinstance(exc, RateLimitError) or status == 429
                if is_rate_limited:
                    n_429 += 1
                if not is_rate_limited or attempt > self.max_retries:
                    self.last_attempts = attempt
                    self.last_n_429 = n_429
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
