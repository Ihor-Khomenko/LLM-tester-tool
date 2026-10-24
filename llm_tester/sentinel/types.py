"""Shared types for MCP-Sentinel decisions (self-contained)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Decision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ASK_USER = "ask_user"
    SANITIZE = "sanitize"
    ESCALATE = "escalate"  # internal: grey zone before S4


class Checkpoint(str, Enum):
    T0 = "T0"  # tool registration
    T1 = "T1"  # tool output
    T2 = "T2"  # before tool call


class Stage(str, Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"
    T0 = "T0"
    CASCADE = "cascade"


@dataclass
class StageScore:
    stage: str
    score: float
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GuardDecision:
    """Result of a sentinel check at T0 / T1 / T2."""

    decision: Decision
    risk_score: float
    stage: str
    checkpoint: str
    reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    stage_scores: List[StageScore] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    escalated: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


@dataclass
class ToolSpec:
    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None

    def canonical(self) -> str:
        schema = self.input_schema or {}
        return f"{self.name}\n{self.description}\n{schema}"


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def verbalize(self) -> str:
        parts = [f"{k}={v!r}" for k, v in sorted(self.arguments.items())]
        return f"{self.name}({', '.join(parts)})"
