"""Build index.csv over a tree of run directories."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_tester.artifacts.run_dir import (
    CONFIG_NAME,
    ENV_NAME,
    METRICS_NAME,
    RESULTS_NAME,
    iter_run_dirs,
    load_config,
    load_metrics,
)

INDEX_COLUMNS = [
    "run_id",
    "path",
    "command",
    "dataset",
    "provider",
    "model",
    "guard",
    "ablation",
    "timestamp_utc",
    "git_sha",
    "n",
    "asr",
    "asr_with_guard",
    "asr_without_guard",
    "tpr",
    "fpr",
    "f1",
    "escalation_rate",
    "refusal_rate",
    "latency_p50_ms",
    "latency_p95_ms",
    "mean_latency_s",
]


def _ablation_label(config: Dict[str, Any]) -> str:
    # Prefer explicit slug written by the CLI.
    explicit = config.get("ablation")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    parts: List[str] = []
    for key in ("no_s1", "no_s2", "no_s3", "no_s4", "only_s4"):
        if config.get(key):
            parts.append(key.replace("_", "-"))
    stages = config.get("stages")
    if isinstance(stages, dict) and not parts:
        # S4 is off by default — only treat S1–S3 as ablation signals here.
        for s in ("s1", "s2", "s3"):
            if stages.get(s) is False:
                parts.append(f"-{s}")
        if stages.get("s4") is True:
            parts.append("+s4")
    return ",".join(parts) if parts else "full"


def _pick(metrics: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in metrics and metrics[k] is not None:
            return metrics[k]
    return None


def summarise_run(run_dir: Path) -> Dict[str, Any]:
    """Flatten one run directory into an index row."""
    config: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    env: Dict[str, Any] = {}
    try:
        config = load_config(run_dir)
    except FileNotFoundError:
        pass
    try:
        metrics = load_metrics(run_dir)
    except FileNotFoundError:
        pass
    env_path = Path(run_dir) / ENV_NAME
    if env_path.exists():
        import json

        env = json.loads(env_path.read_text(encoding="utf-8"))

    asr = _pick(metrics, "attack_success_rate", "asr", "asr_with_guard")
    row = {
        "run_id": Path(run_dir).name,
        "path": str(Path(run_dir).resolve()),
        "command": config.get("command") or env.get("command") or "",
        "dataset": config.get("dataset") or metrics.get("dataset") or "",
        "provider": config.get("provider")
        or env.get("provider")
        or metrics.get("provider")
        or "",
        "model": config.get("model") or env.get("model") or metrics.get("model") or "",
        "guard": config.get("guard")
        or ("sentinel" if config.get("with_guard") else config.get("guard", "")),
        "ablation": _ablation_label(config),
        "timestamp_utc": env.get("timestamp_utc") or "",
        "git_sha": (env.get("git") or {}).get("sha") or "",
        "n": _pick(metrics, "n", "total", "evaluated"),
        "asr": asr,
        "asr_with_guard": _pick(metrics, "asr_with_guard"),
        "asr_without_guard": _pick(metrics, "asr_without_guard"),
        "tpr": _pick(metrics, "tpr"),
        "fpr": _pick(metrics, "fpr"),
        "f1": _pick(metrics, "f1"),
        "escalation_rate": _pick(metrics, "escalation_rate"),
        "refusal_rate": _pick(metrics, "refusal_rate"),
        "latency_p50_ms": _pick(metrics, "latency_p50_ms"),
        "latency_p95_ms": _pick(metrics, "latency_p95_ms"),
        "mean_latency_s": _pick(metrics, "mean_latency_s"),
        "has_results": (Path(run_dir) / RESULTS_NAME).exists(),
        "has_config": (Path(run_dir) / CONFIG_NAME).exists(),
        "has_metrics": (Path(run_dir) / METRICS_NAME).exists(),
    }
    return row


def build_index(runs_root: Path) -> List[Dict[str, Any]]:
    rows = [summarise_run(p) for p in iter_run_dirs(Path(runs_root))]
    return rows


def write_index_csv(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Path:
    cols = list(columns or INDEX_COLUMNS)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})
    return out_path
