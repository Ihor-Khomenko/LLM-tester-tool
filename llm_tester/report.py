"""Summarise metrics from JSONL (or legacy JSON) result files."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


@dataclass
class ReportStats:
    files: List[str] = field(default_factory=list)
    total: int = 0
    errors: int = 0
    skipped: int = 0
    blocked: int = 0
    refusals: int = 0
    attack_successes: int = 0
    latencies: List[float] = field(default_factory=list)
    by_provider_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def evaluated(self) -> int:
        return max(0, self.total - self.errors - self.skipped)

    @property
    def refusal_rate(self) -> Optional[float]:
        n = self.evaluated
        return (self.refusals / n) if n else None

    @property
    def attack_success_rate(self) -> Optional[float]:
        n = self.evaluated
        return (self.attack_successes / n) if n else None

    @property
    def mean_latency_s(self) -> Optional[float]:
        return (sum(self.latencies) / len(self.latencies)) if self.latencies else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files": self.files,
            "total": self.total,
            "evaluated": self.evaluated,
            "errors": self.errors,
            "skipped": self.skipped,
            "blocked": self.blocked,
            "refusals": self.refusals,
            "attack_successes": self.attack_successes,
            "refusal_rate": self.refusal_rate,
            "attack_success_rate": self.attack_success_rate,
            "mean_latency_s": self.mean_latency_s,
            "by_provider_model": self.by_provider_model,
        }


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    def _from_jsonl(src: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for line in src.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    if path.name.endswith(".jsonl.gz") or path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = fh.read().strip()
        if text:
            yield from _from_jsonl(text)
        return

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    if path.suffix == ".jsonl":
        yield from _from_jsonl(text)
        return

    # Heuristic: multi-line file where each non-empty line is a JSON object
    if "\n" in text and text.lstrip().startswith("{"):
        try:
            rows = _from_jsonl(text)
            if rows:
                yield from rows
                return
        except json.JSONDecodeError:
            pass

    data = json.loads(text)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        yield data


def _bucket_key(rec: Dict[str, Any]) -> str:
    provider = rec.get("provider") or "unknown"
    model = rec.get("model") or "unknown"
    return f"{provider}/{model}"


def _accumulate(stats: ReportStats, rec: Dict[str, Any]) -> None:
    stats.total += 1
    status = (rec.get("status") or "").lower()
    latency = rec.get("latency_s")
    if isinstance(latency, (int, float)):
        stats.latencies.append(float(latency))

    key = _bucket_key(rec)
    bucket = stats.by_provider_model.setdefault(
        key,
        {
            "total": 0,
            "refusals": 0,
            "attack_successes": 0,
            "errors": 0,
            "blocked": 0,
            "latencies": [],
        },
    )
    bucket["total"] += 1
    if isinstance(latency, (int, float)):
        bucket["latencies"].append(float(latency))

    if status == "error" or rec.get("error"):
        stats.errors += 1
        bucket["errors"] += 1
        return
    if status == "skipped":
        stats.skipped += 1
        return

    refused = rec.get("refused")
    attack = rec.get("attack_success")
    guard_blocked = bool(rec.get("guard_blocked")) or status == "blocked"

    # Legacy failure-only JSON: status=failed means non-refusal
    if refused is None and status == "failed":
        refused = False
        attack = True

    if guard_blocked:
        stats.blocked += 1
        bucket["blocked"] += 1

    if refused is True:
        stats.refusals += 1
        bucket["refusals"] += 1
    elif refused is False or attack is True:
        stats.attack_successes += 1
        bucket["attack_successes"] += 1


def summarise(paths: Sequence[Union[str, Path]]) -> ReportStats:
    stats = ReportStats()
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Results file not found: {path}")
        stats.files.append(str(path))
        for rec in _iter_records(path):
            _accumulate(stats, rec)

    # Finalise per-bucket rates
    for bucket in stats.by_provider_model.values():
        n = bucket["total"] - bucket["errors"]
        bucket["refusal_rate"] = (bucket["refusals"] / n) if n else None
        bucket["attack_success_rate"] = (
            (bucket["attack_successes"] / n) if n else None
        )
        lats = bucket.pop("latencies")
        bucket["mean_latency_s"] = (sum(lats) / len(lats)) if lats else None
    return stats


def format_report(stats: ReportStats) -> str:
    lines = [
        "=== LLM Tester Report ===",
        f"Files: {', '.join(stats.files) or '(none)'}",
        f"Total records: {stats.total}",
        f"Evaluated: {stats.evaluated}",
        f"Errors: {stats.errors}",
        f"Blocked (guard): {stats.blocked}",
        f"Refusals: {stats.refusals}",
        f"Attack successes: {stats.attack_successes}",
    ]
    if stats.refusal_rate is not None:
        lines.append(f"Refusal rate: {stats.refusal_rate:.2%}")
    if stats.attack_success_rate is not None:
        lines.append(f"Attack success rate: {stats.attack_success_rate:.2%}")
    if stats.mean_latency_s is not None:
        lines.append(f"Mean latency: {stats.mean_latency_s:.3f}s")
    if stats.by_provider_model:
        lines.append("")
        lines.append("By provider/model:")
        for key, b in sorted(stats.by_provider_model.items()):
            rr = b.get("refusal_rate")
            asr = b.get("attack_success_rate")
            ml = b.get("mean_latency_s")
            rr_s = f"{rr:.2%}" if rr is not None else "n/a"
            asr_s = f"{asr:.2%}" if asr is not None else "n/a"
            ml_s = f"{ml:.3f}s" if ml is not None else "n/a"
            lines.append(
                f"  {key}: n={b['total']} refusal={rr_s} asr={asr_s} latency={ml_s}"
            )
    return "\n".join(lines)
