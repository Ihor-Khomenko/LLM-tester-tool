"""YAML config for ``llmtest sentinel-proxy``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

DEFAULT_THRESHOLD = 0.7


@dataclass
class ProxyConfig:
    """Runtime settings for the MCP proxy (stdio or HTTP)."""

    threshold: float = DEFAULT_THRESHOLD  # → policy.tau_high
    tau_low: float = 0.25
    enable_s4: bool = False
    s4_provider: Optional[str] = None
    s4_model: Optional[str] = None
    user_task: str = ""
    audit_log: Optional[str] = None
    policy: Optional[str] = None
    transport: str = "stdio"  # stdio | http
    listen: str = "127.0.0.1:8765"
    backend: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_proxy_config(
    path: Optional[Union[str, Path]] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> ProxyConfig:
    """Load proxy YAML; missing file → defaults. ``overrides`` win."""
    data: Dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if p.is_file():
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Proxy config must be a mapping: {p}")
            data.update(loaded)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v

    known = {f.name for f in ProxyConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known}
    cfg = ProxyConfig(**filtered)
    transport = (cfg.transport or "stdio").lower().strip()
    if transport not in ("stdio", "http"):
        raise ValueError(f"Unknown transport {cfg.transport!r}; use stdio|http")
    cfg.transport = transport
    return cfg


def apply_proxy_config_to_policy(policy: Any, cfg: ProxyConfig) -> Any:
    """Mutate / return a SentinelPolicy with proxy threshold + S4 flags."""
    policy.tau_high = float(cfg.threshold)
    policy.tau_low = float(cfg.tau_low)
    policy.enable_s4 = bool(cfg.enable_s4)
    if cfg.s4_model:
        policy.s4_model = cfg.s4_model
    if cfg.s4_provider:
        policy.s4_provider = cfg.s4_provider
    return policy
