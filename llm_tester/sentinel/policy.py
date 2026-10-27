"""YAML policy / thresholds for MCP-Sentinel."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_tester.sentinel.side_effects import DEFAULT_SIDE_EFFECT_VERBS

DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "sentinel" / "policy.default.yaml"
)

# Sensitive sink argument names (S2 weights). Aligned with the concept doc:
# recipient / url / path / amount / command (+ close variants). Generic
# body/content fields stay at default weight so summarisation is not FP'd.
DEFAULT_SENSITIVE_ARGS = (
    "recipient",
    "to",
    "email",
    "url",
    "uri",
    "href",
    "path",
    "filepath",
    "filename",
    "command",
    "cmd",
    "amount",
    "iban",
    "account",
    "password",
    "token",
    "api_key",
    "guest",
    "guest_id",
    "destination",
    "address",
    "phone",
    "wire_to",
    "payee",
)

DEFAULT_DESTRUCTIVE_TOOLS = (
    "delete",
    "remove",
    "rm",
    "transfer",
    "send_money",
    "wire",
    "grant",
    "revoke",
    "exec",
    "shell",
    "run_command",
)


@dataclass
class SentinelPolicy:
    """Configurable cascade thresholds and stage toggles."""

    # Cascade thresholds on calibrated risk r in [0, 1].
    tau_low: float = 0.25
    tau_high: float = 0.70

    # How enabled stage scores combine: noisy_or (monotone) | max | weighted.
    risk_aggregation: str = "noisy_or"

    # Stage enable flags (ablation).
    enable_s1: bool = True
    enable_s2: bool = True
    enable_s3: bool = True
    enable_s4: bool = False  # optional / expensive; off by default
    # Ablation: S4-only mode (S1–S3 off). When True, every T2 call is sent to
    # the S4 judge and its score alone decides vs ``tau_high`` (default 0.7).
    # Set by :meth:`with_ablation` ``only_s4=True`` — not a free-standing toggle.
    only_s4: bool = False
    # Checkpoint enable flags (ablation: with/without T0 / T1).
    enable_t0: bool = True
    enable_t1: bool = True

    # Stage weights kept for backwards-compat / reporting; aggregation no longer
    # dilutes via weighted averages of disabled stages.
    weight_s1: float = 0.35
    weight_s2: float = 0.45
    weight_s3: float = 0.20

    # S1
    s1_block_threshold: float = 0.85
    block_destructive_tools: bool = True
    destructive_tool_substrings: List[str] = field(
        default_factory=lambda: list(DEFAULT_DESTRUCTIVE_TOOLS)
    )
    # Unauthorized sensitive-path / credential reads (generic patterns).
    block_sensitive_paths: bool = True
    sensitive_path_score: float = 0.9

    # S2
    s2_block_threshold: float = 0.55
    sensitive_args: List[str] = field(
        default_factory=lambda: list(DEFAULT_SENSITIVE_ARGS)
    )
    sensitive_arg_weight: float = 1.5
    default_arg_weight: float = 0.35  # non-sink args: informational, rarely decisive
    taint_ngram_n: int = 3
    # Generic side-effect *verbs* (name segments / description cues) — not
    # fixture-specific tool ids. Classification also uses tool schema sinks.
    side_effect_verbs: List[str] = field(
        default_factory=lambda: list(DEFAULT_SIDE_EFFECT_VERBS)
    )
    # Deprecated alias kept so older YAML still loads; merged into verbs.
    side_effect_tools: List[str] = field(default_factory=list)

    # S3 — backend: lite | embeddings | llm (llm needs injected LLMChat)
    s3_block_threshold: float = 0.65
    s3_min_similarity: float = 0.25  # below → high intent mismatch
    s3_backend: str = "lite"

    # S4 — concrete client is injected via LLMChat; these are host hints only
    s4_provider: str = "ollama"
    s4_model: Optional[str] = None
    # Judge prompt/parser family: auto | allow_block | llama_guard
    s4_judge_family: str = "auto"
    # Ambiguous judge output: defer (keep prior risk) | allow | block
    s4_ambiguous_policy: str = "defer"
    ask_user_on_block: bool = False

    # T0
    t0_semantic_diff_threshold: float = 0.35
    t0_block_on_hash_mismatch: bool = True

    # Logging
    log_path: Optional[str] = None

    # Action when risk >= tau_high
    high_risk_action: str = "block"  # block | ask_user

    def effective_side_effect_verbs(self) -> List[str]:
        verbs = list(self.side_effect_verbs or [])
        # Legacy YAML key: treat entries as additional verbs/aliases.
        for extra in self.side_effect_tools or []:
            if extra and extra not in verbs:
                verbs.append(extra)
        return verbs

    def stage_enabled(self, stage: str) -> bool:
        key = stage.upper()
        return {
            "S1": self.enable_s1,
            "S2": self.enable_s2,
            "S3": self.enable_s3,
            "S4": self.enable_s4,
        }.get(key, True)

    def with_ablation(
        self,
        *,
        no_s1: bool = False,
        no_s2: bool = False,
        no_s3: bool = False,
        no_s4: bool = False,
        only_s4: bool = False,
    ) -> "SentinelPolicy":
        p = SentinelPolicy(**{f.name: getattr(self, f.name) for f in fields(self)})
        if only_s4:
            p.enable_s1 = False
            p.enable_s2 = False
            p.enable_s3 = False
            p.enable_s4 = True
            p.only_s4 = True
            return p
        # Other ablations must not leave a stale only_s4 flag from a copy.
        p.only_s4 = False
        if no_s1:
            p.enable_s1 = False
        if no_s2:
            p.enable_s2 = False
        if no_s3:
            p.enable_s3 = False
        if no_s4:
            p.enable_s4 = False
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentinelPolicy":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**kwargs)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for sentinel policy files. "
            "Install with: pip install PyYAML"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Policy file must be a mapping: {path}")
    return raw


def load_policy(path: Optional[Path] = None) -> SentinelPolicy:
    """Load policy from YAML, falling back to built-in defaults."""
    if path is None:
        if DEFAULT_POLICY_PATH.is_file():
            return SentinelPolicy.from_dict(_load_yaml(DEFAULT_POLICY_PATH))
        return SentinelPolicy()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy file not found: {path}")
    return SentinelPolicy.from_dict(_load_yaml(path))


def merge_policy(base: SentinelPolicy, overrides: Dict[str, Any]) -> SentinelPolicy:
    data = base.to_dict()
    data.update({k: v for k, v in overrides.items() if v is not None})
    return SentinelPolicy.from_dict(data)


def is_sensitive_arg(name: str, sensitive: Sequence[str]) -> bool:
    key = name.lower().strip()
    for s in sensitive:
        s_l = s.lower()
        if key == s_l or s_l in key or key.endswith(s_l):
            return True
    return False
