"""Dataset loaders for HarmfulQA and ToxicChat."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_tester.config import default_dataset_path


@dataclass
class PromptItem:
    """A single evaluable prompt with optional metadata."""

    id: str
    prompt: str
    dataset: str
    type: str = "question"
    category: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def load_harmfulqa(
    path: Optional[Path] = None,
    *,
    expand_conversations: bool = False,
) -> List[PromptItem]:
    """Load HarmfulQA prompts.

    By default only the top-level ``question`` field is used (matches the
    original unprotected scripts). With ``expand_conversations=True``, also
    emit every human turn from blue/red conversations (matches the original
    LLM Guard script's single-prompt pass).
    """
    file_path = Path(path) if path else default_dataset_path("harmfulqa")
    if not file_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {file_path}.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[PromptItem] = []
    for entry in data:
        entry_id = str(entry.get("id", len(prompts) + 1))
        topic = entry.get("topic", "")
        subtopic = entry.get("subtopic", "")
        category = f"{topic}/{subtopic}" if topic or subtopic else None
        question = entry.get("question")
        if question:
            prompts.append(
                PromptItem(
                    id=f"{entry_id}_question",
                    prompt=question,
                    dataset="harmfulqa",
                    type="question",
                    category=category,
                    meta={"entry_id": entry_id, "topic": topic, "subtopic": subtopic},
                )
            )
        if not expand_conversations:
            continue
        for conv_id, conv in entry.get("blue_conversations", {}).items():
            for i, turn in enumerate(conv):
                if turn.get("from") == "human" and turn.get("value"):
                    prompts.append(
                        PromptItem(
                            id=f"{entry_id}_blue_{conv_id}_{i}",
                            prompt=turn["value"],
                            dataset="harmfulqa",
                            type="blue_conversation",
                            category=f"{category}/blue_{conv_id}" if category else None,
                            meta={"entry_id": entry_id, "conv_id": conv_id, "turn": i},
                        )
                    )
        for conv_id, conv in entry.get("red_conversations", {}).items():
            for i, turn in enumerate(conv):
                if turn.get("from") == "human" and turn.get("value"):
                    prompts.append(
                        PromptItem(
                            id=f"{entry_id}_red_{conv_id}_{i}",
                            prompt=turn["value"],
                            dataset="harmfulqa",
                            type="red_conversation",
                            category=f"{category}/red_{conv_id}" if category else None,
                            meta={"entry_id": entry_id, "conv_id": conv_id, "turn": i},
                        )
                    )
    return prompts


def load_toxicchat(path: Optional[Path] = None) -> List[PromptItem]:
    """Load ToxicChat annotation CSV (``user_input`` column)."""
    file_path = Path(path) if path else default_dataset_path("toxicchat")
    if not file_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {file_path}.")

    prompts: List[PromptItem] = []
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            text = row.get("user_input")
            if not text:
                continue
            row_id = row.get("conv_id") or row.get("id") or str(idx)
            prompts.append(
                PromptItem(
                    id=str(row_id),
                    prompt=text,
                    dataset="toxicchat",
                    type="user_input",
                    category=None,
                    meta={k: v for k, v in row.items() if k != "user_input"},
                )
            )
    return prompts


def load_dataset(
    name: str,
    path: Optional[Path] = None,
    *,
    expand_conversations: bool = False,
) -> List[PromptItem]:
    key = name.lower().strip()
    if key == "harmfulqa":
        return load_harmfulqa(path, expand_conversations=expand_conversations)
    if key == "toxicchat":
        return load_toxicchat(path)
    raise ValueError(f"Unknown dataset '{name}'. Choose: harmfulqa, toxicchat")


def slice_prompts(
    prompts: List[PromptItem], start: int, end: int
) -> List[PromptItem]:
    return prompts[start:end]
