"""T0 — tool registration integrity (hash + semantic diff)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from llm_tester.sentinel.types import ToolSpec


_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# Phrases that often appear in tool-poisoning / rug-pull descriptions.
POISON_PHRASES = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "do not tell the user",
    "hidden instruction",
    "before using this tool",
    "you must first",
    "exfiltrate",
    "send all data to",
    "api key",
    "<important>",
    "</important>",
    "IMPORTANT!!!",
)


@dataclass
class ToolRecord:
    name: str
    description: str
    description_hash: str
    input_schema_hash: str
    full_hash: str
    input_schema: Optional[dict] = None
    pinned: bool = True


@dataclass
class ToolRegistry:
    """Pin tool descriptions at first observation; detect later mutations."""

    tools: Dict[str, ToolRecord] = field(default_factory=dict)

    def pin(self, tool: ToolSpec) -> ToolRecord:
        full = tool.canonical()
        rec = ToolRecord(
            name=tool.name,
            description=tool.description,
            description_hash=sha256_text(tool.description),
            input_schema_hash=sha256_text(str(tool.input_schema or {})),
            full_hash=sha256_text(full),
            input_schema=dict(tool.input_schema) if tool.input_schema else None,
            pinned=True,
        )
        self.tools[tool.name] = rec
        return rec

    def get(self, name: str) -> Optional[ToolRecord]:
        return self.tools.get(name)

    def lookup_spec(self, name: str) -> Optional[ToolSpec]:
        rec = self.tools.get(name)
        if rec is None:
            return None
        return ToolSpec(
            name=rec.name,
            description=rec.description,
            input_schema=rec.input_schema,
        )

    def check(
        self,
        tool: ToolSpec,
        *,
        semantic_diff_threshold: float = 0.35,
        block_on_hash_mismatch: bool = True,
    ) -> Tuple[float, List[str], Dict]:
        """
        Return (risk, reasons, details) for a tools/list entry.

        First sighting pins the hash (risk 0 unless poison phrases found).
        Later mismatches raise risk (rug pull / tool poisoning).
        """
        reasons: List[str] = []
        details: Dict = {}
        risk = 0.0

        poison_hits = [p for p in POISON_PHRASES if p.lower() in tool.description.lower()]
        if poison_hits:
            risk = max(risk, 0.95)
            reasons.append(f"poison_phrase:{poison_hits[0]}")
            details["poison_phrases"] = poison_hits

        existing = self.tools.get(tool.name)
        full_hash = sha256_text(tool.canonical())
        details["hash"] = full_hash

        if existing is None:
            self.pin(tool)
            details["action"] = "pinned"
            return risk, reasons, details

        details["pinned_hash"] = existing.full_hash
        if full_hash != existing.full_hash:
            details["action"] = "hash_mismatch"
            reasons.append("tool_description_hash_mismatch")
            # Semantic distance: 1 - jaccard of description tokens.
            sim = jaccard(_tokens(existing.description), _tokens(tool.description))
            dist = 1.0 - sim
            details["semantic_distance"] = dist
            details["token_similarity"] = sim
            if dist >= semantic_diff_threshold:
                reasons.append(f"semantic_diff:{dist:.2f}")
                risk = max(risk, min(1.0, 0.5 + dist))
            else:
                risk = max(risk, 0.4)
            if block_on_hash_mismatch:
                risk = max(risk, 0.8)
        else:
            details["action"] = "unchanged"

        return risk, reasons, details

    def register_many(self, tools: Iterable[ToolSpec]) -> List[ToolRecord]:
        return [self.pin(t) for t in tools]
