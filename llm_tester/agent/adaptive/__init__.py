"""Adaptive attacks targeting S2 attribution (paraphrase, split, launder, mimic)."""

from llm_tester.agent.adaptive.generate import (
    ATTACK_KINDS,
    generate_adaptive_dataset,
    load_adaptive_traces,
    mutate_trace,
)

__all__ = [
    "ATTACK_KINDS",
    "generate_adaptive_dataset",
    "load_adaptive_traces",
    "mutate_trace",
]
