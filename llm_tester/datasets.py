"""Dataset registry and loaders (local files + Hugging Face, cached)."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for ``llmtest datasets`` and the loader registry."""

    name: str
    description: str
    source: str
    hf_id: Optional[str] = None
    hf_config: Optional[str] = None
    hf_split: Optional[str] = None
    local_fallback: bool = False
    tags: tuple = ()


def cache_dir() -> Path:
    """Local cache root for Hugging Face downloads."""
    override = os.environ.get("LLMTEST_CACHE_DIR")
    if override:
        path = Path(override)
    else:
        path = Path.cwd() / ".cache" / "llm_tester"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hf_cache_dir() -> Path:
    path = cache_dir() / "hf"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for Hugging Face loaders. "
            "Install with: pip install 'llm-tester[datasets]' or pip install datasets"
        ) from exc
    return load_dataset


def _load_hf(
    hf_id: str,
    *,
    config: Optional[str] = None,
    split: Optional[str] = None,
) -> Sequence[Dict[str, Any]]:
    """Load a HF dataset split into a list of row dicts (cached under .cache/)."""
    load_dataset = _require_datasets()
    kwargs: Dict[str, Any] = {
        "cache_dir": str(hf_cache_dir()),
    }
    # Honour offline mode for CI / air-gapped runs once cache is warm.
    if os.environ.get("HF_DATASETS_OFFLINE", "").strip() in ("1", "true", "yes"):
        kwargs["download_mode"] = "reuse_dataset_if_exists"

    if config:
        ds = load_dataset(hf_id, config, **kwargs)
    else:
        ds = load_dataset(hf_id, **kwargs)

    if hasattr(ds, "keys"):
        # DatasetDict
        split_name = split
        if split_name is None:
            for candidate in ("test", "train", "harmful", "validation"):
                if candidate in ds:
                    split_name = candidate
                    break
            if split_name is None:
                split_name = next(iter(ds.keys()))
        rows = ds[split_name]
    else:
        rows = ds

    return [dict(row) for row in rows]


def _first_str(row: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            if not val:
                continue
            val = val[0]
        text = str(val).strip()
        if text:
            return text
    return None


# ---------------------------------------------------------------------------
# Local / hybrid loaders
# ---------------------------------------------------------------------------


def load_harmfulqa(
    path: Optional[Path] = None,
    *,
    expand_conversations: bool = False,
) -> List[PromptItem]:
    """Load HarmfulQA prompts from local JSON (or HF if local missing)."""
    file_path = Path(path) if path else default_dataset_path("harmfulqa")
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # declare-lab/HarmfulQA on HF
        data = list(_load_hf("declare-lab/HarmfulQA", split="train"))
        if not data:
            raise FileNotFoundError(
                f"Test dataset not found at {file_path} and HF load returned empty."
            )

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
        for conv_id, conv in (entry.get("blue_conversations") or {}).items():
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
        for conv_id, conv in (entry.get("red_conversations") or {}).items():
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
    """Load ToxicChat from local CSV or Hugging Face ``lmsys/toxic-chat``."""
    file_path = Path(path) if path else default_dataset_path("toxicchat")
    prompts: List[PromptItem] = []

    if file_path.exists():
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

    # HF mirror (toxicchat-0124-test config when available)
    try:
        rows = _load_hf("lmsys/toxic-chat", config="toxicchat0124", split="test")
    except Exception:
        rows = _load_hf("lmsys/toxic-chat", split="test")

    for idx, row in enumerate(rows):
        text = _first_str(row, ("user_input", "user", "prompt", "text"))
        if not text:
            continue
        row_id = str(row.get("conv_id") or row.get("id") or idx)
        prompts.append(
            PromptItem(
                id=row_id,
                prompt=text,
                dataset="toxicchat",
                type="user_input",
                meta={k: v for k, v in row.items() if k not in ("user_input", "user", "prompt", "text")},
            )
        )
    return prompts


def load_advbench(path: Optional[Path] = None) -> List[PromptItem]:
    """AdvBench harmful behaviors (``walledai/AdvBench`` / goal column)."""
    if path:
        return _load_csv_goals(path, dataset="advbench", prompt_keys=("goal", "prompt", "behavior"))

    rows = _load_hf("walledai/AdvBench", split="train")
    prompts: List[PromptItem] = []
    for idx, row in enumerate(rows):
        text = _first_str(row, ("prompt", "goal", "behavior", "content"))
        if not text:
            continue
        prompts.append(
            PromptItem(
                id=f"advbench_{idx}",
                prompt=text,
                dataset="advbench",
                type="harmful_behavior",
                category=_first_str(row, ("category", "Category")),
                meta={k: v for k, v in row.items() if k not in ("prompt", "goal")},
            )
        )
    return prompts


def load_harmbench(path: Optional[Path] = None) -> List[PromptItem]:
    """HarmBench standard behaviors (``walledai/HarmBench``, config ``standard``)."""
    if path:
        return _load_csv_goals(
            path,
            dataset="harmbench",
            prompt_keys=("Behavior", "prompt", "goal", "behavior"),
        )

    rows = _load_hf("walledai/HarmBench", config="standard", split="train")
    prompts: List[PromptItem] = []
    for idx, row in enumerate(rows):
        text = _first_str(row, ("prompt", "Behavior", "behavior", "goal"))
        if not text:
            continue
        category = _first_str(row, ("category", "Category", "SemanticCategory"))
        prompts.append(
            PromptItem(
                id=f"harmbench_{idx}",
                prompt=text,
                dataset="harmbench",
                type="behavior",
                category=category,
                meta=dict(row),
            )
        )
    return prompts


def load_jailbreakbench(path: Optional[Path] = None) -> List[PromptItem]:
    """JailbreakBench JBB-Behaviors harmful split."""
    if path:
        return _load_csv_goals(
            path,
            dataset="jailbreakbench",
            prompt_keys=("Goal", "goal", "prompt", "Behavior"),
        )

    rows = _load_hf("JailbreakBench/JBB-Behaviors", config="behaviors", split="harmful")
    prompts: List[PromptItem] = []
    for idx, row in enumerate(rows):
        text = _first_str(row, ("Goal", "goal", "prompt"))
        if not text:
            continue
        index = row.get("Index", idx)
        category = _first_str(row, ("Category", "category"))
        prompts.append(
            PromptItem(
                id=f"jbb_{index}",
                prompt=text,
                dataset="jailbreakbench",
                type="harmful_behavior",
                category=category,
                meta=dict(row),
            )
        )
    return prompts


def load_prompt_injections(
    path: Optional[Path] = None,
    *,
    injections_only: bool = True,
) -> List[PromptItem]:
    """deepset/prompt-injections — verified HF dataset for prompt-injection text.

    By default only rows with ``label == 1`` (injection) are returned.
    """
    if path:
        rows_local: List[Dict[str, Any]] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows_local = list(reader)
        rows: Sequence[Dict[str, Any]] = rows_local
    else:
        # Concatenate train+test for a full evaluation set.
        train = _load_hf("deepset/prompt-injections", split="train")
        test = _load_hf("deepset/prompt-injections", split="test")
        rows = list(train) + list(test)

    prompts: List[PromptItem] = []
    for idx, row in enumerate(rows):
        label = row.get("label")
        try:
            label_int = int(label) if label is not None and str(label) != "" else None
        except (TypeError, ValueError):
            label_int = None
        if injections_only and label_int is not None and label_int != 1:
            continue
        text = _first_str(row, ("text", "prompt", "user_input"))
        if not text:
            continue
        prompts.append(
            PromptItem(
                id=f"pinject_{idx}",
                prompt=text,
                dataset="promptinject",
                type="injection" if label_int == 1 else "benign",
                category="prompt_injection" if label_int == 1 else "benign",
                meta={"label": label_int},
            )
        )
    return prompts


def _load_csv_goals(
    path: Path,
    *,
    dataset: str,
    prompt_keys: Sequence[str],
) -> List[PromptItem]:
    prompts: List[PromptItem] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            text = _first_str(row, prompt_keys)
            if not text:
                continue
            prompts.append(
                PromptItem(
                    id=f"{dataset}_{idx}",
                    prompt=text,
                    dataset=dataset,
                    type="behavior",
                    category=_first_str(row, ("category", "Category")),
                    meta=dict(row),
                )
            )
    return prompts


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LoaderFn = Callable[..., List[PromptItem]]

DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "harmfulqa": DatasetInfo(
        name="harmfulqa",
        description="HarmfulQA questions (local harmfuldataset.json or declare-lab/HarmfulQA)",
        source="local or Hugging Face",
        hf_id="declare-lab/HarmfulQA",
        local_fallback=True,
        tags=("jailbreak", "harmful"),
    ),
    "toxicchat": DatasetInfo(
        name="toxicchat",
        description="ToxicChat user inputs (local CSV or lmsys/toxic-chat)",
        source="local or Hugging Face",
        hf_id="lmsys/toxic-chat",
        local_fallback=True,
        tags=("toxicity",),
    ),
    "advbench": DatasetInfo(
        name="advbench",
        description="AdvBench harmful_behaviors (walledai/AdvBench)",
        source="Hugging Face",
        hf_id="walledai/AdvBench",
        hf_split="train",
        tags=("jailbreak", "harmful"),
    ),
    "harmbench": DatasetInfo(
        name="harmbench",
        description="HarmBench standard behaviors (walledai/HarmBench)",
        source="Hugging Face",
        hf_id="walledai/HarmBench",
        hf_config="standard",
        hf_split="train",
        tags=("jailbreak", "harmful"),
    ),
    "jailbreakbench": DatasetInfo(
        name="jailbreakbench",
        description="JailbreakBench JBB-Behaviors harmful split",
        source="Hugging Face",
        hf_id="JailbreakBench/JBB-Behaviors",
        hf_config="behaviors",
        hf_split="harmful",
        tags=("jailbreak", "harmful"),
    ),
    "promptinject": DatasetInfo(
        name="promptinject",
        description="deepset/prompt-injections (label=1 injection prompts)",
        source="Hugging Face",
        hf_id="deepset/prompt-injections",
        tags=("prompt-injection",),
    ),
}

# Aliases for convenience
DATASET_ALIASES: Dict[str, str] = {
    "jbb": "jailbreakbench",
    "jbb-behaviors": "jailbreakbench",
    "prompt-injections": "promptinject",
    "prompt_injections": "promptinject",
    "advbench_harmful_behaviors": "advbench",
}


def resolve_dataset_name(name: str) -> str:
    key = name.lower().strip()
    key = DATASET_ALIASES.get(key, key)
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {known}")
    return key


def list_datasets() -> List[DatasetInfo]:
    return [DATASET_REGISTRY[k] for k in sorted(DATASET_REGISTRY)]


def format_dataset_list() -> str:
    lines = ["Available datasets:", ""]
    for info in list_datasets():
        hf = f"  HF: {info.hf_id}" if info.hf_id else ""
        if info.hf_config:
            hf += f" ({info.hf_config})"
        lines.append(f"  {info.name}")
        lines.append(f"    {info.description}")
        lines.append(f"    source: {info.source}{hf}")
        if info.tags:
            lines.append(f"    tags: {', '.join(info.tags)}")
        lines.append("")
    lines.append(f"HF cache directory: {hf_cache_dir()}")
    return "\n".join(lines)


def load_dataset(
    name: str,
    path: Optional[Path] = None,
    *,
    expand_conversations: bool = False,
) -> List[PromptItem]:
    key = resolve_dataset_name(name)
    if key == "harmfulqa":
        return load_harmfulqa(path, expand_conversations=expand_conversations)
    if key == "toxicchat":
        return load_toxicchat(path)
    if key == "advbench":
        return load_advbench(path)
    if key == "harmbench":
        return load_harmbench(path)
    if key == "jailbreakbench":
        return load_jailbreakbench(path)
    if key == "promptinject":
        return load_prompt_injections(path)
    raise ValueError(f"No loader for dataset '{key}'")


def slice_prompts(
    prompts: List[PromptItem], start: int, end: int
) -> List[PromptItem]:
    return prompts[start:end]


def dataset_choices() -> List[str]:
    return sorted(DATASET_REGISTRY.keys())
