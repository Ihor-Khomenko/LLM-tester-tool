#!/usr/bin/env python3
"""Legacy entry point — HarmfulQA with LLM Guard + local Ollama.

Prefer the new CLI:
  llmtest run --dataset harmfulqa --provider ollama --model llama3 \\
    --guard llmguard --expand-conversations --range ...
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_tester.config import RunParams
from llm_tester.datasets import load_harmfulqa, slice_prompts
from llm_tester.guard import build_guard
from llm_tester.judges import PhraseJudge
from llm_tester.providers import LLMProvider
from llm_tester.range_utils import interactive_select_range
from llm_tester.range_utils import parse_range_spec as parse_range_input  # noqa: F401
from llm_tester.runner import print_summary, run_evaluation


def main() -> None:
    prompts = load_harmfulqa(
        Path("harmfuldataset.json"),
        expand_conversations=True,
    )
    start, end = interactive_select_range(len(prompts), label="scenarios")
    selected = slice_prompts(prompts, start, end)

    provider = LLMProvider("ollama", "llama3", params=RunParams())
    guard = build_guard("llmguard")
    out = Path("guarded_results.jsonl")
    summary = run_evaluation(
        selected,
        provider,
        judge=PhraseJudge(),
        guard=guard,
        out_path=out,
        verbose=True,
        start_number=start + 1,
    )
    print_summary(summary, out)
    _write_legacy_csv(out, Path("guarded_results.csv"))


def _write_legacy_csv(jsonl_path: Path, csv_path: Path) -> None:
    fieldnames = [
        "id",
        "type",
        "category",
        "prompt",
        "response",
        "input_scores",
        "output_scores",
        "input_scan_time",
        "query_time",
        "model",
    ]
    rows = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "id": rec.get("id"),
                    "type": rec.get("type"),
                    "category": rec.get("category"),
                    "prompt": rec.get("prompt"),
                    "response": rec.get("response"),
                    "input_scores": str(rec.get("input_scores") or {}),
                    "output_scores": str(rec.get("output_scores") or {}),
                    "input_scan_time": rec.get("input_scan_time") or 0,
                    "query_time": rec.get("latency_s") or 0,
                    "model": "guarded",
                }
            )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Legacy CSV written to {csv_path}")


if __name__ == "__main__":
    main()
