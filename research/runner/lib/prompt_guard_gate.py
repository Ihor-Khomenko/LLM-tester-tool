"""PromptGuard 2 baseline gate for the live research runner.

Wraps ``llm_tester.baselines.prompt_guard.PromptGuard2Guard`` (Meta
Llama-Prompt-Guard-2-22M via transformers, threshold 0.5). Weights are never
bundled: set ``PROMPT_GUARD_MODEL_PATH`` (or rely on the HF cache) for offline
loads. If ``transformers`` / torch / weights are missing, the guard skips
gracefully (ALLOW + ``available=False``) so CI and hosts without the optional
dependency stay green. Never downloads in CI (respects HF_HUB_OFFLINE /
TRANSFORMERS_OFFLINE).
"""
from __future__ import annotations

import os
from typing import Any, Optional

# Re-export env name used by the baseline implementation.
ENV_MODEL_PATH = "PROMPT_GUARD_MODEL_PATH"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-22M"


def build_prompt_guard2(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model_id: Optional[str] = None,
    classifier: Any = None,
):
    """Return a PromptGuard2Guard. ``classifier`` injects a mock score fn (tests)."""
    from llm_tester.baselines.prompt_guard import PromptGuard2Guard

    mid = model_id or os.environ.get(ENV_MODEL_PATH) or DEFAULT_MODEL_ID
    return PromptGuard2Guard(
        model_id=mid,
        threshold=float(threshold),
        classifier=classifier,
    )


def is_prompt_guard_arm(name: Optional[str]) -> bool:
    n = (name or "").lower().strip().replace("-", "_")
    return n in ("prompt_guard2", "prompt_guard", "promptguard2", "promptguard", "pg2")
