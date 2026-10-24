"""Core evaluation loop — query, judge, and log every response as JSONL."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO

from llm_tester.datasets import PromptItem
from llm_tester.guard import NoGuard
from llm_tester.judges import Judge, JudgeResult, PhraseJudge
from llm_tester.providers import LLMProvider


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TestRecord:
    test_number: int
    id: str
    prompt: str
    response: Optional[str]
    refused: Optional[bool]
    attack_success: Optional[bool]
    status: str
    judge: str
    judge_label: Optional[str] = None
    judge_detail: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    latency_s: Optional[float] = None
    guard: str = "none"
    guard_blocked: bool = False
    input_scores: Dict[str, Any] = field(default_factory=dict)
    output_scores: Dict[str, Any] = field(default_factory=dict)
    input_scan_time: Optional[float] = None
    output_scan_time: Optional[float] = None
    dataset: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    total: int = 0
    skipped: int = 0
    resumed_skipped: int = 0
    errors: int = 0
    refusals: int = 0
    attack_successes: int = 0
    blocked: int = 0
    requests: int = 0
    stopped_max_requests: bool = False
    latencies: List[float] = field(default_factory=list)

    @property
    def evaluated(self) -> int:
        return self.total - self.skipped - self.errors - self.resumed_skipped

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


class JSONLWriter:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh: TextIO = open(self.path, mode, encoding="utf-8")

    def write(self, record: TestRecord) -> None:
        self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_completed_ids(path: Path) -> Set[str]:
    """Return prompt ids already present in a JSONL results file."""
    done: Set[str] = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id")
            if pid is not None:
                done.add(str(pid))
    return done


def run_evaluation(
    prompts: Iterable[PromptItem],
    provider: LLMProvider,
    *,
    judge: Optional[Judge] = None,
    guard=None,
    out_path: Optional[Path] = None,
    verbose: bool = True,
    start_number: int = 1,
    resume: bool = False,
    max_requests: Optional[int] = None,
    delay: float = 0.0,
) -> RunSummary:
    """Run prompts against provider; log every response to JSONL when out_path set.

    When ``resume`` is True and ``out_path`` exists, prompt ids already logged are
    skipped and new rows are appended. ``max_requests`` caps new provider calls
    so daily quotas can be spread across runs. ``delay`` sleeps between calls.
    """
    judge = judge or PhraseJudge()
    guard = guard or NoGuard()
    summary = RunSummary()
    out = Path(out_path) if out_path else None
    completed: Set[str] = load_completed_ids(out) if resume and out else set()
    writer: Optional[JSONLWriter] = None
    if out:
        writer = JSONLWriter(out, append=bool(resume and out.exists()))
    info = provider.info()
    made_request = False

    try:
        for offset, item in enumerate(prompts):
            test_number = start_number + offset
            if not item.prompt:
                summary.skipped += 1
                summary.total += 1
                if verbose:
                    print(f"Skipping item {test_number}: empty prompt.")
                continue

            summary.total += 1

            if item.id in completed:
                summary.resumed_skipped += 1
                if verbose:
                    print(f"Resume skip {test_number}: id={item.id} already in {out}")
                continue

            if max_requests is not None and summary.requests >= max_requests:
                summary.stopped_max_requests = True
                if verbose:
                    print(
                        f"Reached --max-requests={max_requests}; "
                        f"stopping (resume later with the same --out)."
                    )
                break

            if delay > 0 and made_request:
                time.sleep(delay)

            record = TestRecord(
                test_number=test_number,
                id=item.id,
                prompt=item.prompt,
                response=None,
                refused=None,
                attack_success=None,
                status="pending",
                judge=getattr(judge, "name", "unknown"),
                provider=info["provider"],
                model=info["model"],
                base_url=info["base_url"],
                params=dict(info["params"]),
                guard=getattr(guard, "name", "none"),
                dataset=item.dataset,
                type=item.type,
                category=item.category,
            )

            try:
                g_in = guard.scan_input(item.prompt)
                record.input_scores = g_in.input_scores
                record.input_scan_time = g_in.input_scan_time

                if not g_in.allowed:
                    record.response = g_in.response_override or "Blocked"
                    record.guard_blocked = True
                    record.refused = True
                    record.attack_success = False
                    record.status = "blocked"
                    record.judge_label = "blocked"
                    summary.blocked += 1
                    summary.refusals += 1
                else:
                    chat = provider.chat(g_in.sanitized_prompt)
                    summary.requests += 1
                    made_request = True
                    record.latency_s = chat.latency_s
                    summary.latencies.append(chat.latency_s)

                    g_out = guard.scan_output(g_in.sanitized_prompt, chat.content)
                    record.output_scores = g_out.output_scores
                    record.output_scan_time = g_out.output_scan_time

                    if not g_out.allowed:
                        record.response = g_out.response_override or "Blocked"
                        record.guard_blocked = True
                        record.refused = True
                        record.attack_success = False
                        record.status = "blocked"
                        record.judge_label = "blocked"
                        summary.blocked += 1
                        summary.refusals += 1
                    else:
                        record.response = g_out.sanitized_output or chat.content
                        verdict: JudgeResult = judge.judge(
                            item.prompt, record.response or ""
                        )
                        record.refused = verdict.refused
                        record.attack_success = not verdict.refused
                        record.judge_label = verdict.label
                        record.judge_detail = verdict.detail
                        record.status = "refusal" if verdict.refused else "attack_success"
                        if verdict.refused:
                            summary.refusals += 1
                        else:
                            summary.attack_successes += 1

                if verbose:
                    print(f"Test {test_number}: Prompt: {item.prompt}")
                    print(f"Response: {record.response}\n")

            except Exception as exc:  # noqa: BLE001 — research harness logs all failures
                record.error = str(exc)
                record.status = "error"
                record.response = None
                summary.errors += 1
                if verbose:
                    print(f"Error querying model: {exc}\n")

            if writer:
                writer.write(record)
                completed.add(item.id)
    finally:
        if writer:
            writer.close()

    return summary


def print_summary(summary: RunSummary, out_path: Optional[Path] = None) -> None:
    print(f"\nSkipped (empty): {summary.skipped}")
    print(f"Skipped (resume): {summary.resumed_skipped}")
    print(f"Provider requests: {summary.requests}")
    print(f"Errors: {summary.errors}")
    print(f"Blocked (guard): {summary.blocked}")
    print(f"Refusals: {summary.refusals}")
    print(f"Attack successes: {summary.attack_successes}")
    if summary.refusal_rate is not None:
        print(f"Refusal rate: {summary.refusal_rate:.2%}")
    if summary.attack_success_rate is not None:
        print(f"Attack success rate: {summary.attack_success_rate:.2%}")
    if summary.mean_latency_s is not None:
        print(f"Mean latency: {summary.mean_latency_s:.3f}s")
    if summary.stopped_max_requests:
        print("Stopped early due to --max-requests (re-run with --resume to continue).")
    if out_path:
        print(f"All results logged to {out_path}")
