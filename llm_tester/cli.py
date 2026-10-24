"""Console entry point: ``llmtest run``, ``report``, and ``datasets``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from llm_tester.config import (
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    PROVIDERS,
    RunParams,
)
from llm_tester.datasets import (
    dataset_choices,
    format_dataset_list,
    load_dataset,
    slice_prompts,
)
from llm_tester.guard import build_guard
from llm_tester.judges import build_judge
from llm_tester.providers import LLMProvider
from llm_tester.range_utils import parse_range_spec
from llm_tester.report import format_report, summarise
from llm_tester.runner import print_summary, run_evaluation


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=dataset_choices(),
        help="Dataset to evaluate (see: llmtest datasets)",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Override path to dataset file (skips Hugging Face download)",
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=sorted(PROVIDERS.keys()),
        help="LLM provider (OpenAI-compatible)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id (default depends on provider; OpenRouter free models use :free)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override provider base URL",
    )
    parser.add_argument(
        "--guard",
        default="none",
        choices=["none", "llmguard"],
        help="Input/output guard mode (default: none)",
    )
    parser.add_argument(
        "--judge",
        default="phrase",
        choices=["phrase", "llamaguard", "llm"],
        help="Refusal detection strategy (default: phrase)",
    )
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=sorted(PROVIDERS.keys()),
        help="Provider for llamaguard/llm judges (default: same as --provider)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for llamaguard/llm judges",
    )
    parser.add_argument(
        "--range",
        dest="range_spec",
        default="all",
        help="Prompt range: all|N|start-end (default: all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results.jsonl"),
        help="JSONL output path (every response is logged)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to --out and skip prompt ids already present (quota-friendly)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after N new provider calls (combine with --resume across days)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between provider requests (rate-limit friendliness)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries on HTTP 429 with exponential backoff (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed for deterministic runs (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not send a seed parameter",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens for completions",
    )
    parser.add_argument(
        "--expand-conversations",
        action="store_true",
        help="For harmfulqa: also test human turns from blue/red conversations",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-prompt stdout (still writes JSONL)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtest",
        description="Test LLM refusal behaviour on harmful prompts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an evaluation")
    _add_run_args(run_p)

    report_p = sub.add_parser("report", help="Summarise metrics from results files")
    report_p.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more JSONL/JSON results files",
    )
    report_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )

    sub.add_parser("datasets", help="List available datasets and sources")
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    prompts = load_dataset(
        args.dataset,
        args.dataset_path,
        expand_conversations=args.expand_conversations,
    )
    if not prompts:
        print("No prompts loaded.", file=sys.stderr)
        return 1

    parsed = parse_range_spec(args.range_spec, len(prompts))
    if parsed is None:
        print(
            f"Invalid --range '{args.range_spec}' for dataset of length {len(prompts)}.",
            file=sys.stderr,
        )
        return 2
    start, end = parsed
    selected = slice_prompts(prompts, start, end)
    print(
        f"Dataset={args.dataset} prompts={len(selected)} "
        f"(indices {start}:{end}) provider={args.provider} "
        f"guard={args.guard} judge={args.judge}"
    )
    if args.resume:
        print(f"Resume enabled: skipping ids already in {args.out}")
    if args.max_requests is not None:
        print(f"Max new requests this run: {args.max_requests}")

    params = RunParams(
        temperature=args.temperature,
        seed=None if args.no_seed else args.seed,
        max_tokens=args.max_tokens,
    )
    provider = LLMProvider(
        args.provider,
        args.model,
        base_url=args.base_url,
        params=params,
        max_retries=args.max_retries,
    )

    judge_provider = None
    if args.judge in ("llamaguard", "llm"):
        j_provider_name = args.judge_provider or args.provider
        j_model = args.judge_model
        if args.judge == "llamaguard" and not j_model:
            j_model = "llama-guard3"
        judge_provider = LLMProvider(
            j_provider_name,
            j_model,
            params=params,
            max_retries=args.max_retries,
        )
    judge = build_judge(args.judge, judge_provider=judge_provider)
    guard = build_guard(args.guard)

    summary = run_evaluation(
        selected,
        provider,
        judge=judge,
        guard=guard,
        out_path=args.out,
        verbose=not args.quiet,
        start_number=start + 1,
        resume=args.resume,
        max_requests=args.max_requests,
        delay=args.delay,
    )
    print_summary(summary, args.out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    stats = summarise(args.files)
    if args.json:
        print(json.dumps(stats.to_dict(), indent=2))
    else:
        print(format_report(stats))
    return 0


def cmd_datasets(_args: argparse.Namespace) -> int:
    print(format_dataset_list())
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "datasets":
        return cmd_datasets(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
