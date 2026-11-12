"""External detector baseline gates for the live / detector research runner.

Wraps PromptGuard 2, ProtectAI DeBERTa, Llama Guard 3, and Qwen3Guard-Gen.
Weights are never bundled; missing / gated models skip gracefully
(``available=False``, ALLOW) so CI stays offline. Llama Guard gated-access
failures include an explicit HF license / HF_TOKEN message.
"""
from __future__ import annotations

from typing import Any, Optional, Set

# Re-export PromptGuard helpers for back-compat imports.
from prompt_guard_gate import (  # noqa: F401
    DEFAULT_MODEL_ID as PROMPT_GUARD_DEFAULT_MODEL_ID,
    DEFAULT_THRESHOLD as PROMPT_GUARD_DEFAULT_THRESHOLD,
    ENV_MODEL_PATH as PROMPT_GUARD_ENV_MODEL_PATH,
    build_prompt_guard2,
    is_prompt_guard_arm,
)

DEFAULT_THRESHOLD = 0.5

EXTERNAL_DETECTOR_ARMS: Set[str] = {
    "prompt_guard2",
    "prompt_guard",
    "promptguard2",
    "promptguard",
    "pg2",
    "protectai_deberta",
    "protectai",
    "deberta_pi",
    "deberta_pi_v2",
    "llama_guard3",
    "llama_guard",
    "llamaguard3",
    "lg3",
    "qwen3_guard",
    "qwen3guard",
    "qwen_guard",
}


def canonical_detector_arm(name: Optional[str]) -> Optional[str]:
    n = (name or "").lower().strip().replace("-", "_")
    if n in ("prompt_guard2", "prompt_guard", "promptguard2", "promptguard", "pg2"):
        return "prompt_guard2"
    if n in ("protectai_deberta", "protectai", "deberta_pi", "deberta_pi_v2", "protectai_deberta_v2"):
        return "protectai_deberta"
    if n in ("llama_guard3", "llama_guard", "llamaguard3", "llamaguard", "lg3"):
        return "llama_guard3"
    if n in ("qwen3_guard", "qwen3guard", "qwen_guard", "qwen3_guard_gen"):
        return "qwen3_guard"
    return None


def is_external_detector_arm(name: Optional[str]) -> bool:
    return canonical_detector_arm(name) is not None


def build_detector_guard(
    name: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model_id: Optional[str] = None,
    classifier: Any = None,
):
    """Build a GuardLike for an external detector arm (or raise ValueError)."""
    canon = canonical_detector_arm(name)
    if canon is None:
        raise ValueError(f"Not an external detector arm: {name!r}")
    if canon == "prompt_guard2":
        return build_prompt_guard2(
            threshold=threshold, model_id=model_id, classifier=classifier
        )
    from llm_tester.baselines.registry import build_baseline

    return build_baseline(
        canon,
        detector_model=model_id,
        detector_threshold=threshold,
        detector_classifier=classifier,
    )
