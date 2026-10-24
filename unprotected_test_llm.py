#!/usr/bin/env python3
"""Legacy entry point — HarmfulQA against local Ollama (no guard).

Prefer the new CLI:
  llmtest run --dataset harmfulqa --provider ollama --model llama3 --range ...
"""

from __future__ import annotations

from pathlib import Path

from llm_tester.config import RunParams
from llm_tester.datasets import load_harmfulqa, slice_prompts
from llm_tester.judges import PhraseJudge
from llm_tester.providers import LLMProvider
from llm_tester.range_utils import interactive_select_range
from llm_tester.runner import print_summary, run_evaluation

# Re-export helpers so older imports / customisations keep working.
from llm_tester.judges import is_safe_refusal  # noqa: F401
from llm_tester.range_utils import parse_range_spec as parse_range_input  # noqa: F401


def query_clean(prompt: str) -> str:
    provider = LLMProvider("ollama", "llama3", params=RunParams())
    return provider.chat(prompt).content


def main() -> None:
    prompts = load_harmfulqa(Path("harmfuldataset.json"))
    start, end = interactive_select_range(len(prompts), label="sets")
    selected = slice_prompts(prompts, start, end)

    provider = LLMProvider("ollama", "llama3", params=RunParams())
    # Legacy scripts only logged failures; the package logs everything.
    # Keep the historical default filename but use JSONL for the full log.
    out = Path("test_results.jsonl")
    summary = run_evaluation(
        selected,
        provider,
        judge=PhraseJudge(),
        out_path=out,
        verbose=True,
        start_number=start + 1,
    )
    print_summary(summary, out)
    # Also write a legacy-style failures-only JSON for reproducibility of old workflows.
    _write_failures_json(out, Path("test_results.json"))


def _write_failures_json(jsonl_path: Path, json_path: Path) -> None:
    import json

    failures = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") in ("attack_success", "error") or rec.get("refused") is False:
                failures.append(
                    {
                        "test_number": rec.get("test_number"),
                        "prompt": rec.get("prompt"),
                        "response": rec.get("response"),
                        "timestamp": rec.get("timestamp"),
                        "status": "failed",
                        **({"error": rec["error"]} if rec.get("error") else {}),
                    }
                )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)
    print(f"Legacy failures-only log written to {json_path}")


if __name__ == "__main__":
    main()
