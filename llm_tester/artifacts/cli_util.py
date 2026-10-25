"""Shared CLI helpers for --run-dir artefact wiring."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_tester.artifacts.run_dir import (
    finalize_run_dir,
    open_results_writer,
    resolve_run_dir,
)
from llm_tester.runner import RunSummary


def add_run_dir_args(parser: argparse.ArgumentParser, *, default_name_help: str) -> None:
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Directory for reproducible artefacts "
            f"(config.yaml, env.json, results.jsonl.gz, metrics.json, README.md). "
            f"Default: runs/<YYYYMMDD>_<name> ({default_name_help})"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Slug used in the default runs/<date>_<name> path",
    )
    parser.add_argument(
        "--extra-sha",
        action="append",
        default=None,
        help=(
            "Optional extra git SHA to record in env.json "
            "(repeatable; e.g. mcp-security-research). Format: label=sha or bare sha"
        ),
    )


def parse_extra_sha(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def summary_to_metrics(summary: RunSummary, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "total": summary.total,
        "evaluated": summary.evaluated,
        "skipped": summary.skipped,
        "resumed_skipped": summary.resumed_skipped,
        "errors": summary.errors,
        "refusals": summary.refusals,
        "attack_successes": summary.attack_successes,
        "blocked": summary.blocked,
        "requests": summary.requests,
        "refusal_rate": summary.refusal_rate,
        "attack_success_rate": summary.attack_success_rate,
        "asr": summary.attack_success_rate,
        "mean_latency_s": summary.mean_latency_s,
        "stopped_max_requests": summary.stopped_max_requests,
    }
    if summary.latencies:
        import statistics

        ordered = sorted(summary.latencies)

        def _pct(p: float) -> float:
            if len(ordered) == 1:
                return float(ordered[0])
            k = (len(ordered) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(ordered) - 1)
            if f == c:
                return float(ordered[f])
            return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))

        d["latency_p50_s"] = _pct(50)
        d["latency_p95_s"] = _pct(95)
        d["latency_mean_s"] = statistics.mean(summary.latencies)
    if extra:
        d.update(extra)
    return d


def agent_summary_to_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    total = int(summary.get("total") or 0)
    n_attack = int(summary.get("n_attack") or 0)
    n_benign = int(summary.get("n_benign") or 0)
    attack_successes = int(summary.get("attack_successes") or 0)
    blocked = int(summary.get("blocked") or 0)
    d = dict(summary)
    d["n"] = total
    d["n_attack"] = n_attack
    d["n_benign"] = n_benign
    # ASR = successful attacks / attack scenarios (not all scenarios).
    d["asr"] = (attack_successes / n_attack) if n_attack else None
    d["attack_success_rate"] = d["asr"]
    d["block_rate"] = (blocked / total) if total else None
    return d


def prepare_run_dir(
    args: argparse.Namespace,
    command: str,
    *,
    name_parts: Sequence[str],
) -> Path:
    name = args.run_name or "_".join(str(p) for p in name_parts if p)
    path = resolve_run_dir(args.run_dir, command, name=name)
    path.mkdir(parents=True, exist_ok=True)
    return path
