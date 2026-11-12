#!/usr/bin/env python3
"""Research-runner driver for detector-only TPR/FPR evaluation.

No Ollama agent loop — classifies benign/attack texts with each defense arm
and writes a JSONL + summary.json under the job outdir. CI / offline: pass
``--corpus synthetic`` (bundled fixtures; no downloads).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python run_detector_eval.py` from research/runner/lib.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        default="synthetic",
        help="synthetic | injecagent | bipia | deepset_prompt_injections",
    )
    ap.add_argument(
        "--arms",
        default="oberih_s1,oberih_s1s2,oberih,prompt_guard2,protectai_deberta,llama_guard3,qwen3_guard",
        help="Comma-separated defense arms",
    )
    ap.add_argument("--langs", default="en", help="Comma-separated language tags for reporting")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--run-tag", default="detector")
    ap.add_argument("--out", type=Path, required=True, help="results JSONL path")
    ap.add_argument("--json-summary", type=Path, default=None)
    args = ap.parse_args(argv)

    from llm_tester.agent.detector_eval import (
        evaluate_detector_items,
        format_detector_report,
        summarize_results,
    )
    from llm_tester.agent.external_datasets import resolve_detector_corpus

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()] or ["en"]

    all_items = []
    for lang in langs:
        items = resolve_detector_corpus(
            args.corpus,
            cache_dir=args.cache_dir,
            split=args.split,
            limit=args.limit,
            language=lang,
        )
        all_items.extend(items)

    results = evaluate_detector_items(
        all_items, arms, threshold=float(args.threshold)
    )
    summary = summarize_results(results)
    summary["run_tag"] = args.run_tag
    summary["corpus"] = args.corpus
    summary["arms"] = arms
    summary["langs"] = langs

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in summary["arms"]:
            row = dict(row)
            row["run_tag"] = args.run_tag
            row["corpus"] = args.corpus
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.json_summary or args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(format_detector_report(summary), flush=True)
    print(f"wrote {args.out} and {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
