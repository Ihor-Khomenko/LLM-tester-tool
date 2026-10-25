"""Factory for paper baselines (+ optional MCP-Sentinel wrapper name)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_tester.baselines.progent import ProgentGuard
from llm_tester.baselines.prompt_guard import PromptGuard2Guard
from llm_tester.baselines.spotlighting import SpotlightingGuard
from llm_tester.baselines.task_shield import TaskShieldGuard

BASELINE_NAMES = (
    "spotlighting",
    "prompt_guard2",
    "progent",
    "task_shield",
    "sentinel",  # MCP-Sentinel via build_sentinel
)


def list_baselines() -> List[str]:
    return list(BASELINE_NAMES)


def build_baseline(
    name: str,
    *,
    llm_provider: Any = None,
    spotlight_mode: str = "datamark",
    prompt_guard_model: Optional[str] = None,
    prompt_guard_classifier: Any = None,
    prompt_guard_threshold: float = 0.5,
    sentinel_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Construct a GuardLike instance.

    ``sentinel`` delegates to ``build_sentinel``. PromptGuard skips when weights
    are missing unless ``prompt_guard_classifier`` is injected (tests).
    """
    key = (name or "").lower().strip().replace("-", "_")
    if key in ("spotlighting", "spotlight"):
        return SpotlightingGuard(mode=spotlight_mode)
    if key in ("prompt_guard2", "prompt_guard", "promptguard2", "promptguard"):
        return PromptGuard2Guard(
            model_id=prompt_guard_model or "meta-llama/Llama-Prompt-Guard-2-22M",
            threshold=prompt_guard_threshold,
            classifier=prompt_guard_classifier,
        )
    if key in ("progent", "privilege", "static_policy"):
        return ProgentGuard()
    if key in ("task_shield", "taskshield", "prompt_armor", "promptarmor"):
        return TaskShieldGuard(provider=llm_provider)
    if key in ("sentinel", "mcp_sentinel", "mcp-sentinel"):
        from llm_tester.sentinel.guard import build_sentinel

        kw = dict(sentinel_kwargs or {})
        if llm_provider is not None and "llm_provider" not in kw:
            kw["llm_provider"] = llm_provider
        return build_sentinel(**kw)
    raise ValueError(
        f"Unknown baseline {name!r}. Choose from: {', '.join(BASELINE_NAMES)}"
    )
