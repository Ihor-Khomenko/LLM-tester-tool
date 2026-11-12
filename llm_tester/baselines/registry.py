"""Factory for paper baselines (+ optional Оберіг / legacy Sentinel wrapper)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_tester.baselines.llama_guard import LlamaGuard3Guard
from llm_tester.baselines.progent import ProgentGuard
from llm_tester.baselines.prompt_guard import PromptGuard2Guard
from llm_tester.baselines.protectai_deberta import ProtectAIDebertaGuard
from llm_tester.baselines.qwen3_guard import Qwen3GuardGenGuard
from llm_tester.baselines.spotlighting import SpotlightingGuard
from llm_tester.baselines.task_shield import TaskShieldGuard
from llm_tester.sentinel.branding import (
    MACHINE_ID,
    is_full_stack_defense,
    is_graded_defense,
    is_oberih_defense,
)

BASELINE_NAMES = (
    "spotlighting",
    "prompt_guard2",
    "protectai_deberta",
    "llama_guard3",
    "qwen3_guard",
    "progent",
    "task_shield",
    MACHINE_ID,  # Оберіг via build_oberih / build_sentinel
    "sentinel",  # legacy alias
)

# Comparison detector arms (not Oberih policy variants).
DETECTOR_BASELINE_NAMES = (
    "prompt_guard2",
    "protectai_deberta",
    "llama_guard3",
    "qwen3_guard",
)


def list_baselines() -> List[str]:
    return list(BASELINE_NAMES)


def list_detector_baselines() -> List[str]:
    return list(DETECTOR_BASELINE_NAMES)


def build_baseline(
    name: str,
    *,
    llm_provider: Any = None,
    spotlight_mode: str = "datamark",
    prompt_guard_model: Optional[str] = None,
    prompt_guard_classifier: Any = None,
    prompt_guard_threshold: float = 0.5,
    detector_model: Optional[str] = None,
    detector_classifier: Any = None,
    detector_threshold: Optional[float] = None,
    sentinel_kwargs: Optional[Dict[str, Any]] = None,
    oberih_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Construct a GuardLike instance.

    ``oberih`` / ``sentinel`` delegate to ``build_oberih`` (alias of
    ``build_sentinel``). Detector baselines skip when weights are missing unless
    ``prompt_guard_classifier`` / ``detector_classifier`` is injected (tests).
    """
    key = (name or "").lower().strip().replace("-", "_")
    thr = (
        float(detector_threshold)
        if detector_threshold is not None
        else float(prompt_guard_threshold)
    )
    clf = detector_classifier if detector_classifier is not None else prompt_guard_classifier

    if key in ("spotlighting", "spotlight"):
        return SpotlightingGuard(mode=spotlight_mode)
    if key in ("prompt_guard2", "prompt_guard", "promptguard2", "promptguard", "pg2"):
        return PromptGuard2Guard(
            model_id=prompt_guard_model
            or detector_model
            or "meta-llama/Llama-Prompt-Guard-2-22M",
            threshold=thr,
            classifier=clf,
        )
    if key in (
        "protectai_deberta",
        "protectai",
        "deberta_pi",
        "deberta_pi_v2",
        "protectai_deberta_v2",
    ):
        return ProtectAIDebertaGuard(
            model_id=detector_model
            or prompt_guard_model
            or "protectai/deberta-v3-base-prompt-injection-v2",
            threshold=thr,
            classifier=clf,
        )
    if key in ("llama_guard3", "llama_guard", "llamaguard3", "llamaguard", "lg3"):
        return LlamaGuard3Guard(
            model_id=detector_model
            or prompt_guard_model
            or "meta-llama/Llama-Guard-3-1B",
            threshold=thr,
            classifier=clf,
        )
    if key in ("qwen3_guard", "qwen3guard", "qwen_guard", "qwen3_guard_gen"):
        return Qwen3GuardGenGuard(
            model_id=detector_model
            or prompt_guard_model
            or "Qwen/Qwen3Guard-Gen-0.6B",
            threshold=thr,
            classifier=clf,
        )
    if key in ("progent", "privilege", "static_policy"):
        return ProgentGuard()
    if key in ("task_shield", "taskshield", "prompt_armor", "promptarmor"):
        return TaskShieldGuard(provider=llm_provider)
    if is_oberih_defense(name) or is_full_stack_defense(name) or key in (
        "oberih",
        "sentinel",
        "mcp_sentinel",
    ):
        from llm_tester.sentinel.guard import build_oberih

        kw = dict(oberih_kwargs or sentinel_kwargs or {})
        if llm_provider is not None and "llm_provider" not in kw:
            kw["llm_provider"] = llm_provider
        guard = build_oberih(**kw)
        if is_graded_defense(name):
            guard.policy.response_mode = "graded"
        return guard
    raise ValueError(
        f"Unknown baseline {name!r}. Choose from: {', '.join(BASELINE_NAMES)}"
    )
