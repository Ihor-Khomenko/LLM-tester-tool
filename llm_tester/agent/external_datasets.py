"""Importers for third-party prompt-injection corpora (detector / FP eval).

Sources (pinned in ``llm_tester/data/external_datasets/manifest.yaml``):

* InjecAgent (uiuc-kang-lab/InjecAgent, MIT) — agentic traces + benign twins
* BIPIA (microsoft/BIPIA, MIT + component licenses) — attack texts + benign contexts
* deepset/prompt-injections (Apache-2.0) — labeled injection / benign text

Large licensed blobs are **not** vendored; use ``scripts/fetch_external_datasets.py``.
CI uses fixtures / mocks only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_tester.agent.traces import (
    SCHEMA_VERSION,
    TraceStep,
    load_injecagent_traces,
)
from llm_tester.sentinel.types import ToolCall, ToolSpec

PACKAGE_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "external_datasets"
    / "manifest.yaml"
)


@dataclass
class DetectorItem:
    """Single text example for detector-only (classification) evaluation."""

    id: str
    text: str
    label: str  # attack | benign
    dataset: str
    language: str = "en"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "dataset": self.dataset,
            "language": self.language,
            "meta": dict(self.meta),
        }


def default_external_cache() -> Path:
    override = os.environ.get("LLMTEST_EXTERNAL_CACHE") or os.environ.get(
        "LLMTEST_CACHE_DIR"
    )
    if override:
        base = Path(override)
        if base.name == "external":
            return base
        return base / "external"
    return Path.cwd() / ".cache" / "llm_tester" / "external"


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or PACKAGE_MANIFEST
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required to read the external dataset manifest") from exc
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _dataset_meta(name: str, manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    man = manifest if manifest is not None else load_manifest()
    meta = ((man.get("datasets") or {}).get(name) or {})
    return {
        "source": meta.get("source"),
        "license": meta.get("license"),
        "version": meta.get("commit") or meta.get("revision"),
        "dataset": name,
    }


# ---------------------------------------------------------------------------
# InjecAgent
# ---------------------------------------------------------------------------


def load_injecagent_detector_items(
    path: Path,
    *,
    split: str = "dh_base",
    limit: Optional[int] = None,
    language: str = "en",
    include_benign_twins: bool = True,
) -> List[DetectorItem]:
    """Flatten InjecAgent cases to text items (tool response = attack surface)."""
    traces = load_injecagent_traces(
        path, split=split, limit=limit, include_benign_twins=include_benign_twins
    )
    pin = _dataset_meta("injecagent")
    items: List[DetectorItem] = []
    for t in traces:
        content = ""
        if t.tool_outputs:
            content = str(t.tool_outputs[0].get("content") or "")
        items.append(
            DetectorItem(
                id=t.id,
                text=content or t.user_request,
                label=t.label,
                dataset="injecagent",
                language=language,
                meta={
                    **pin,
                    **(t.meta or {}),
                    "user_request": t.user_request,
                    "attack_type": t.attack_type,
                },
            )
        )
    return items


def load_injecagent_agentic_traces(
    path: Path,
    *,
    split: str = "dh_base",
    limit: Optional[int] = None,
) -> List[TraceStep]:
    """TraceSteps usable by the live / replay drivers (attack + benign twins)."""
    traces = load_injecagent_traces(path, split=split, limit=limit, include_benign_twins=True)
    pin = _dataset_meta("injecagent")
    for t in traces:
        t.meta = {**(t.meta or {}), **pin}
        t.schema_version = SCHEMA_VERSION
    return traces


# ---------------------------------------------------------------------------
# BIPIA
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _flatten_attack_json(obj: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return (category, attack_text) pairs from BIPIA attack JSON."""
    out: List[Tuple[str, str]] = []
    for cat, vals in (obj or {}).items():
        if isinstance(vals, list):
            for v in vals:
                text = str(v).strip()
                if text:
                    out.append((str(cat), text))
        elif isinstance(vals, dict):
            for sub, inner in vals.items():
                if isinstance(inner, list):
                    for v in inner:
                        text = str(v).strip()
                        if text:
                            out.append((f"{cat}/{sub}", text))
                else:
                    text = str(inner).strip()
                    if text:
                        out.append((f"{cat}/{sub}", text))
        else:
            text = str(vals).strip()
            if text:
                out.append((str(cat), text))
    return out


def load_bipia_detector_items(
    root: Path,
    *,
    split: str = "test",
    tasks: Sequence[str] = ("email", "table", "code"),
    limit: Optional[int] = None,
    language: str = "en",
    include_attacks: bool = True,
    include_benign: bool = True,
) -> List[DetectorItem]:
    """BIPIA attack strings + benign task contexts (third-party FP corpus)."""
    pin = _dataset_meta("bipia")
    items: List[DetectorItem] = []
    n_att = n_ben = 0

    if include_attacks:
        for kind in ("text", "code"):
            path = root / f"{kind}_attack_{split}.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            for cat, text in _flatten_attack_json(data):
                if limit is not None and n_att >= limit:
                    break
                items.append(
                    DetectorItem(
                        id=f"bipia-att-{kind}-{split}-{n_att:04d}",
                        text=text,
                        label="attack",
                        dataset="bipia",
                        language=language,
                        meta={
                            **pin,
                            "split": split,
                            "attack_file": kind,
                            "category": cat,
                            "role": "attack_instruction",
                        },
                    )
                )
                n_att += 1
            if limit is not None and n_att >= limit:
                break

    if include_benign:
        for task in tasks:
            path = root / task / f"{split}.jsonl"
            if not path.is_file():
                continue
            for i, row in enumerate(_iter_jsonl(path)):
                if limit is not None and n_ben >= limit:
                    break
                ctx = str(row.get("context") or row.get("code") or "").strip()
                if not ctx:
                    continue
                items.append(
                    DetectorItem(
                        id=f"bipia-ben-{task}-{split}-{i:04d}",
                        text=ctx,
                        label="benign",
                        dataset="bipia",
                        language=language,
                        meta={
                            **pin,
                            "split": split,
                            "task": task,
                            "question": row.get("question"),
                            "role": "benign_context",
                        },
                    )
                )
                n_ben += 1
            if limit is not None and n_ben >= limit:
                break
    return items


def bipia_agentic_traces(
    root: Path,
    *,
    split: str = "test",
    task: str = "email",
    limit: Optional[int] = None,
) -> List[TraceStep]:
    """Compose BIPIA email/table contexts with injected attacks as agent traces.

    Benign: tool returns clean context; call answers the question.
    Attack: tool returns context + appended attack instruction; call is a sink.
    """
    pin = _dataset_meta("bipia")
    ctx_path = root / task / f"{split}.jsonl"
    att_path = root / (
        f"code_attack_{split}.json" if task == "code" else f"text_attack_{split}.json"
    )
    if not ctx_path.is_file() or not att_path.is_file():
        raise FileNotFoundError(
            f"BIPIA agentic conversion needs {ctx_path} and {att_path}. "
            f"Run: python scripts/fetch_external_datasets.py --datasets bipia"
        )
    attacks = _flatten_attack_json(json.loads(att_path.read_text(encoding="utf-8")))
    tool_name = {
        "email": "email.read",
        "table": "table.query",
        "code": "code.read",
    }.get(task, f"{task}.read")
    sink = ToolCall(
        name="notes.write",
        arguments={"path": "/tmp/exfil.txt", "content": "INJECTED"},
    )
    traces: List[TraceStep] = []
    for i, row in enumerate(_iter_jsonl(ctx_path)):
        if limit is not None and i >= limit:
            break
        ctx = str(row.get("context") or row.get("code") or "")
        question = str(row.get("question") or f"Summarise the {task} content.")
        tools = [
            ToolSpec(name=tool_name, description=f"Read {task} content"),
            ToolSpec(name="notes.write", description="Write a local note"),
        ]
        # Benign twin
        traces.append(
            TraceStep(
                id=f"bipia-ben-{task}-{split}-{i:04d}",
                user_request=question,
                tools=tools,
                tool_outputs=[{"source": tool_name, "content": ctx}],
                call=ToolCall(name=tool_name, arguments={"query": question}),
                label="benign",
                attack_type="benign",
                meta={**pin, "task": task, "split": split, "role": "benign"},
            )
        )
        cat, att = attacks[i % len(attacks)]
        poisoned = f"{ctx}\n\n[EXTERNAL INSTRUCTION]\n{att}"
        traces.append(
            TraceStep(
                id=f"bipia-att-{task}-{split}-{i:04d}",
                user_request=question,
                tools=tools,
                tool_outputs=[{"source": tool_name, "content": poisoned, "untrusted": True}],
                call=sink,
                label="attack",
                attack_type="indirect_injection",
                meta={
                    **pin,
                    "task": task,
                    "split": split,
                    "category": cat,
                    "role": "attack",
                },
            )
        )
    return traces


# ---------------------------------------------------------------------------
# deepset/prompt-injections
# ---------------------------------------------------------------------------


def load_deepset_detector_items(
    *,
    path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    language: str = "en",
    injections_only: bool = False,
) -> List[DetectorItem]:
    """Load deepset prompt-injections as detector items (attack + benign)."""
    pin = _dataset_meta("deepset_prompt_injections")
    rows: List[Dict[str, Any]] = []
    if path is not None:
        p = Path(path)
        if p.is_dir():
            for split in ("train", "test"):
                f = p / f"{split}.jsonl"
                if f.is_file():
                    rows.extend(list(_iter_jsonl(f)))
        elif p.suffix.lower() == ".jsonl":
            rows = list(_iter_jsonl(p))
        else:
            # CSV / reuse datasets.py loader path
            from llm_tester.datasets import load_prompt_injections

            prompts = load_prompt_injections(p, injections_only=injections_only)
            return [
                DetectorItem(
                    id=pr.id,
                    text=pr.prompt,
                    label="attack" if pr.type == "injection" else "benign",
                    dataset="deepset_prompt_injections",
                    language=language,
                    meta={**pin, **(pr.meta or {})},
                )
                for pr in (prompts if limit is None else prompts[:limit])
            ]
    else:
        # Prefer materialised cache from fetch script; else HF (blocked in CI).
        cache = cache_dir or default_external_cache()
        local = cache / "deepset_prompt_injections"
        if (local / "train.jsonl").is_file() or (local / "test.jsonl").is_file():
            return load_deepset_detector_items(
                path=local, limit=limit, language=language, injections_only=injections_only
            )
        from llm_tester.datasets import load_prompt_injections

        prompts = load_prompt_injections(injections_only=injections_only)
        items = [
            DetectorItem(
                id=pr.id,
                text=pr.prompt,
                label="attack" if pr.type == "injection" else "benign",
                dataset="deepset_prompt_injections",
                language=language,
                meta={**pin, **(pr.meta or {})},
            )
            for pr in prompts
        ]
        return items if limit is None else items[:limit]

    items: List[DetectorItem] = []
    for idx, row in enumerate(rows):
        label_raw = row.get("label")
        try:
            label_int = int(label_raw) if label_raw is not None and str(label_raw) != "" else None
        except (TypeError, ValueError):
            label_int = None
        if injections_only and label_int is not None and label_int != 1:
            continue
        text = str(row.get("text") or row.get("prompt") or "").strip()
        if not text:
            continue
        is_att = label_int == 1
        items.append(
            DetectorItem(
                id=f"deepset-{idx:04d}",
                text=text,
                label="attack" if is_att else "benign",
                dataset="deepset_prompt_injections",
                language=language,
                meta={**pin, "label": label_int, "split": row.get("split")},
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def deepset_to_traces(items: Sequence[DetectorItem]) -> List[TraceStep]:
    """Wrap text PI examples as minimal tool-output traces for evaluate_traces."""
    traces: List[TraceStep] = []
    for it in items:
        is_att = it.label == "attack"
        traces.append(
            TraceStep(
                id=it.id,
                user_request="Summarise the retrieved document.",
                tools=[
                    ToolSpec(name="docs.read", description="Read a document"),
                    ToolSpec(name="notes.write", description="Write a note"),
                ],
                tool_outputs=[
                    {
                        "source": "docs.read",
                        "content": it.text,
                        "untrusted": bool(is_att),
                    }
                ],
                call=(
                    ToolCall(
                        name="notes.write",
                        arguments={"path": "/tmp/out.txt", "content": "FOLLOW"},
                    )
                    if is_att
                    else ToolCall(name="docs.read", arguments={"id": "1"})
                ),
                label="attack" if is_att else "benign",
                attack_type="indirect_injection" if is_att else "benign",
                meta={**(it.meta or {}), "source_dataset": it.dataset, "language": it.language},
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Unified resolver
# ---------------------------------------------------------------------------


def resolve_detector_corpus(
    name: str,
    *,
    cache_dir: Optional[Path] = None,
    path: Optional[Path] = None,
    split: str = "test",
    limit: Optional[int] = None,
    language: str = "en",
) -> List[DetectorItem]:
    """Load a named detector corpus (injecagent | bipia | deepset | synthetic)."""
    key = name.lower().strip().replace("-", "_")
    cache = cache_dir or default_external_cache()

    if key in ("synthetic", "synth"):
        from llm_tester.agent.traces import load_synthetic_traces

        traces = load_synthetic_traces()
        if limit is not None:
            traces = traces[:limit]
        items: List[DetectorItem] = []
        for t in traces:
            content = ""
            if t.tool_outputs:
                content = str(t.tool_outputs[0].get("content") or "")
            items.append(
                DetectorItem(
                    id=t.id,
                    text=content or t.user_request,
                    label=t.label,
                    dataset="synthetic",
                    language=language,
                    meta={"attack_type": t.attack_type, **(t.meta or {})},
                )
            )
        return items

    if key.startswith("injec"):
        local = path
        if local is None:
            cand = cache / "injecagent" / "test_cases_dh_base.json"
            if cand.is_file():
                local = cand
            else:
                raise FileNotFoundError(
                    "InjecAgent cache missing. Run: "
                    "python scripts/fetch_external_datasets.py --datasets injecagent"
                )
        return load_injecagent_detector_items(
            local, split=split if split != "test" else "dh_base", limit=limit, language=language
        )

    if key in ("bipia", "bipia_benign", "bipia_fp"):
        root = path or (cache / "bipia")
        include_att = key != "bipia_benign"
        include_ben = key != "bipia" or True
        if key == "bipia_benign":
            include_att = False
            include_ben = True
        return load_bipia_detector_items(
            Path(root),
            split=split,
            limit=limit,
            language=language,
            include_attacks=include_att,
            include_benign=include_ben,
        )

    if key in ("deepset", "deepset_prompt_injections", "promptinject", "prompt_injections"):
        return load_deepset_detector_items(
            path=path,
            cache_dir=cache,
            limit=limit,
            language=language,
            injections_only=False,
        )

    raise ValueError(
        f"Unknown detector corpus {name!r}. "
        f"Use synthetic|injecagent|bipia|deepset_prompt_injections"
    )


def resolve_agentic_traces(
    name: str,
    *,
    cache_dir: Optional[Path] = None,
    path: Optional[Path] = None,
    split: str = "test",
    limit: Optional[int] = None,
) -> List[TraceStep]:
    """Load agentic TraceSteps where the dataset supports them."""
    key = name.lower().strip().replace("-", "_")
    cache = cache_dir or default_external_cache()
    if key.startswith("injec"):
        local = path or (cache / "injecagent" / "test_cases_dh_base.json")
        return load_injecagent_agentic_traces(Path(local), limit=limit)
    if key in ("bipia",):
        return bipia_agentic_traces(
            Path(path or (cache / "bipia")), split=split, task="email", limit=limit
        )
    if key in ("deepset", "deepset_prompt_injections", "promptinject"):
        items = load_deepset_detector_items(
            path=path, cache_dir=cache, limit=limit, injections_only=False
        )
        return deepset_to_traces(items)
    if key in ("synthetic", "synth"):
        from llm_tester.agent.traces import load_synthetic_traces

        traces = load_synthetic_traces()
        return traces[:limit] if limit is not None else traces
    raise ValueError(f"No agentic conversion for corpus {name!r}")
