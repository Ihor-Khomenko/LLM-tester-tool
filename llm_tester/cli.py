"""Console entry point: ``llmtest run``, ``report``, ``datasets``, ``guard-eval``, ``agent-bench``, ``index``, ``compare``, ``figures``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_tester.artifacts.cli_util import (
    add_run_dir_args,
    agent_summary_to_metrics,
    parse_extra_sha,
    prepare_run_dir,
    summary_to_metrics,
)
from llm_tester.artifacts.compare import compare_runs, format_compare
from llm_tester.artifacts.index import build_index, write_index_csv
from llm_tester.artifacts.run_dir import finalize_run_dir, open_results_writer
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
        default=None,
        help=(
            "Optional legacy/plain JSONL mirror path. "
            "Primary results always go to <run-dir>/results.jsonl.gz"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to run-dir results and skip prompt ids already present (quota-friendly)",
    )
    add_run_dir_args(parser, default_name_help="from dataset/provider/model")
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

    ge = sub.add_parser(
        "guard-eval",
        help="Evaluate MCP-Sentinel on traces (synthetic / InjecAgent / external JSONL)",
    )
    ge.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "injecagent", "jsonl"],
        help="synthetic (offline) | injecagent | jsonl (mcp-sentinel-trace/v1)",
    )
    ge.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to mcp-sentinel-trace JSONL or InjecAgent JSON",
    )
    ge.add_argument(
        "--download-injecagent",
        action="store_true",
        help="Download InjecAgent dh_base JSON from GitHub into .cache/ (needs network)",
    )
    ge.add_argument("--limit", type=int, default=None, help="Max cases (before twins)")
    ge.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Sentinel YAML policy path",
    )
    ge.add_argument(
        "--no-s1",
        action="store_true",
        help="Ablation: disable S1 rules",
    )
    ge.add_argument("--no-s2", action="store_true", help="Ablation: disable S2 taint")
    ge.add_argument("--no-s3", action="store_true", help="Ablation: disable S3 intent")
    ge.add_argument("--no-s4", action="store_true", help="Ablation: disable S4 judge")
    ge.add_argument(
        "--only-s4",
        action="store_true",
        help="Ablation: only S4 (requires --provider)",
    )
    ge.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS.keys()),
        help="Optional LLM for S3(llm)/S4 (ollama|openai|xai|openrouter|groq)",
    )
    ge.add_argument("--model", default=None, help="Model id (OpenRouter :free supported)")
    ge.add_argument("--base-url", default=None, help="Override provider base URL")
    ge.add_argument(
        "--s3-backend",
        default=None,
        choices=["lite", "embeddings", "llm"],
        help="Override policy s3_backend (llm needs --provider)",
    )
    ge.add_argument(
        "--enable-s4",
        action="store_true",
        help="Enable S4 LLM judge (grey zone); needs --provider",
    )
    ge.add_argument(
        "--json",
        action="store_true",
        help="Emit metrics as JSON",
    )
    ge.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional JSONL decision log path",
    )
    ge.add_argument("--max-retries", type=int, default=5)
    ge.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ge.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ge.add_argument("--no-seed", action="store_true")
    add_run_dir_args(ge, default_name_help="guard-eval_<dataset>")

    ab = sub.add_parser(
        "agent-bench",
        help="Scaffold: tool-using agent with/without MCP-Sentinel (any provider)",
    )
    ab.add_argument(
        "--provider",
        required=True,
        choices=sorted(PROVIDERS.keys()),
        help="LLM provider: ollama|openai|xai|openrouter|groq",
    )
    ab.add_argument(
        "--model",
        default=None,
        help="Model id (OpenRouter free tier: suffix :free)",
    )
    ab.add_argument("--base-url", default=None, help="Override provider base URL")
    ab.add_argument(
        "--with-guard",
        action="store_true",
        default=True,
        help="Enable MCP-Sentinel (default)",
    )
    ab.add_argument(
        "--without-guard",
        action="store_true",
        help="Run agent without sentinel",
    )
    ab.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Sentinel YAML policy",
    )
    ab.add_argument(
        "--s3-backend",
        default=None,
        choices=["lite", "embeddings", "llm"],
        help="S3 backend when guard is enabled",
    )
    ab.add_argument(
        "--enable-s4",
        action="store_true",
        help="Enable S4 judge using the same --provider",
    )
    ab.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional legacy/plain JSONL mirror; primary is <run-dir>/results.jsonl.gz",
    )
    ab.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenario ids already present in run-dir results",
    )
    add_run_dir_args(ab, default_name_help="agent-bench_<provider>")
    ab.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Cap new provider calls this run",
    )
    ab.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between provider requests (rate-limit friendliness)",
    )
    ab.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries on HTTP 429 (default 1 — keep free-tier quota)",
    )
    ab.add_argument(
        "--suite",
        default="builtin",
        choices=["builtin", "msr", "jsonl", "injecagent", "agentdojo", "all"],
        help="Scenario suite (agentdojo is a scaffold; msr/jsonl need --scenarios-path)",
    )
    ab.add_argument(
        "--scenarios-path",
        type=Path,
        default=None,
        help="JSONL path for suite=msr|jsonl|all (v1 or mcp-security-research 1.0)",
    )
    ab.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of scenarios loaded",
    )
    ab.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    ab.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ab.add_argument("--no-seed", action="store_true")

    sp = sub.add_parser(
        "sentinel-proxy",
        help="Run MCP stdio JSON-RPC proxy wrapping a backend MCP server",
    )
    sp.add_argument("--policy", type=Path, default=None)
    sp.add_argument("--log", type=Path, default=None)
    sp.add_argument("--user-task", default="")
    sp.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Backend command after --",
    )

    idx = sub.add_parser(
        "index",
        help="Build index.csv over a runs/ tree for thesis experiments",
    )
    idx.add_argument(
        "runs_root",
        type=Path,
        help="Root directory containing run artefact folders",
    )
    idx.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <runs-root>/index.csv)",
    )

    cmp_p = sub.add_parser(
        "compare",
        help="Diff metrics.json across two or more run directories",
    )
    cmp_p.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories (at least two recommended)",
    )
    cmp_p.add_argument(
        "--format",
        dest="fmt",
        default="md",
        choices=["md", "csv", "docx-ready"],
        help="Output format (docx-ready = markdown with Ukrainian table caption above)",
    )
    cmp_p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional file to write (default: stdout)",
    )
    cmp_p.add_argument(
        "--table-number",
        type=int,
        default=1,
        help="Suggested table number for caption (Таблиця 4.n)",
    )

    fig = sub.add_parser(
        "figures",
        help="Emit thesis-ready B/W figures (PNG+SVG), tables, and captions manifest",
    )
    fig.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="One or more run directories (or a runs root with a single child)",
    )
    fig.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for figures / tables / figures_manifest.json",
    )
    fig.add_argument(
        "--lang",
        default="uk",
        choices=["uk", "en"],
        help="Axis labels and suggested captions (default: uk)",
    )
    fig.add_argument(
        "--figure-start",
        type=int,
        default=1,
        help="First figure number for Рисунок 4.n captions",
    )
    fig.add_argument(
        "--table-start",
        type=int,
        default=1,
        help="First table number for Таблиця 4.n captions",
    )
    return parser


def cmd_guard_eval(args: argparse.Namespace) -> int:
    from llm_tester.agent.eval import evaluate_traces, format_metrics
    from llm_tester.agent.provider_adapter import build_llm_chat
    from llm_tester.agent.traces import TraceSchemaError, resolve_traces
    from llm_tester.sentinel.guard import build_sentinel
    from llm_tester.sentinel.policy import load_policy

    try:
        traces = resolve_traces(
            args.dataset,
            path=args.dataset_path,
            limit=args.limit,
            download=args.download_injecagent,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except TraceSchemaError as exc:
        print(f"Trace schema error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Failed to load traces: {exc}", file=sys.stderr)
        return 1

    if not traces:
        print("No traces loaded.", file=sys.stderr)
        return 1

    policy = load_policy(args.policy)
    policy = policy.with_ablation(
        no_s1=args.no_s1,
        no_s2=args.no_s2,
        no_s3=args.no_s3,
        no_s4=args.no_s4,
        only_s4=args.only_s4,
    )
    if args.s3_backend:
        policy.s3_backend = args.s3_backend

    llm = None
    want_llm = (
        args.only_s4
        or args.enable_s4
        or (args.s3_backend == "llm")
        or (policy.s3_backend == "llm")
    )
    if args.only_s4 or args.enable_s4:
        policy.enable_s4 = True
    elif not args.only_s4:
        # Default offline path: no S4 unless explicitly enabled.
        if not args.enable_s4:
            policy.enable_s4 = False

    if want_llm and args.provider:
        llm = build_llm_chat(
            args.provider,
            args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=None if args.no_seed else args.seed,
            max_retries=args.max_retries,
        )
    elif want_llm and not args.provider:
        print(
            "Warning: S3/S4 LLM backend requested but --provider not set; "
            "falling back to lite S3 / no S4.",
            file=sys.stderr,
        )
        if policy.s3_backend == "llm":
            policy.s3_backend = "lite"
        policy.enable_s4 = False

    abl = []
    if args.no_s1:
        abl.append("-S1")
    if args.no_s2:
        abl.append("-S2")
    if args.no_s3:
        abl.append("-S3")
    if args.only_s4:
        abl.append("only-S4")
    abl_slug = "".join(abl) if abl else "full"

    run_dir = prepare_run_dir(
        args,
        "guard-eval",
        name_parts=["guard-eval", args.dataset, abl_slug],
    )
    # Prefer run-dir log when --log not set.
    log_path = args.log
    if log_path is None:
        log_path = run_dir / "sentinel_decisions.jsonl"

    guard = build_sentinel(policy=policy, llm_provider=llm, log_path=log_path)
    metrics, records = evaluate_traces(traces, guard, collect_records=True)

    with open_results_writer(run_dir, append=False) as writer:
        for rec in records:
            writer.write(rec)

    metrics_dict = metrics.to_dict()
    metrics_dict["dataset"] = args.dataset
    metrics_dict["command"] = "guard-eval"
    config: Dict[str, Any] = {
        "command": "guard-eval",
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "limit": args.limit,
        "policy": str(args.policy) if args.policy else None,
        "no_s1": args.no_s1,
        "no_s2": args.no_s2,
        "no_s3": args.no_s3,
        "no_s4": args.no_s4,
        "only_s4": args.only_s4,
        "enable_s4": bool(policy.enable_s4),
        "s3_backend": policy.s3_backend,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "seed": None if args.no_seed else args.seed,
        "thresholds": {
            "tau_low": policy.tau_low,
            "tau_high": policy.tau_high,
            "s1_block_threshold": policy.s1_block_threshold,
        },
        "stages": {
            "s1": policy.enable_s1,
            "s2": policy.enable_s2,
            "s3": policy.enable_s3,
            "s4": policy.enable_s4,
        },
        "ablation": abl_slug,
    }
    finalize_run_dir(
        run_dir,
        command="guard-eval",
        config=config,
        metrics=metrics_dict,
        argv=sys.argv[1:],
        provider=args.provider,
        model=args.model,
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    print(f"Run artefacts written to {run_dir}")

    if args.json:
        print(json.dumps(metrics_dict, indent=2))
    else:
        title = f"guard-eval dataset={args.dataset}" + (
            f" ablation={','.join(abl)}" if abl else ""
        )
        print(format_metrics(metrics, title=title))
        print(
            "\nNote: numbers are measured on this run's traces (preliminary). "
            "Not claimed as published results."
        )
    return 0


def cmd_agent_bench(args: argparse.Namespace) -> int:
    from llm_tester.agent.agent_bench import load_scenarios, run_agent_bench
    from llm_tester.agent.provider_adapter import build_llm_chat
    from llm_tester.agent.traces import TraceSchemaError
    from llm_tester.sentinel.guard import build_sentinel
    from llm_tester.sentinel.policy import load_policy

    try:
        scenarios = load_scenarios(
            suite=args.suite,
            path=args.scenarios_path,
            limit=args.limit,
            include_builtin=args.suite in ("builtin", "default", "all"),
        )
    except (FileNotFoundError, TraceSchemaError, ValueError, NotImplementedError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not scenarios:
        print("No scenarios loaded.", file=sys.stderr)
        return 1

    provider = build_llm_chat(
        args.provider,
        args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=None if args.no_seed else args.seed,
        max_retries=args.max_retries,
    )
    with_guard = not args.without_guard
    guard = None
    if with_guard and (args.enable_s4 or args.s3_backend):
        policy = load_policy(args.policy)
        if args.s3_backend:
            policy.s3_backend = args.s3_backend
        if args.enable_s4:
            policy.enable_s4 = True
        guard = build_sentinel(policy=policy, llm_provider=provider)

    print(
        f"agent-bench suite={args.suite} scenarios={len(scenarios)} "
        f"guard={with_guard} delay={args.delay}s max_retries={args.max_retries}",
        file=sys.stderr,
    )
    run_dir = prepare_run_dir(
        args,
        "agent-bench",
        name_parts=[
            "agent-bench",
            args.provider,
            args.model or "default",
            "guard" if with_guard else "raw",
        ],
    )
    results_gz = run_dir / "results.jsonl.gz"
    # Agent bench currently writes plain JSONL; use a working file then gzip via writer.
    work_jsonl = run_dir / "_results_work.jsonl"
    if args.resume and results_gz.exists() and not work_jsonl.exists():
        # Inflate for resume compatibility with existing agent-bench loader.
        from llm_tester.artifacts.run_dir import load_results

        with work_jsonl.open("w", encoding="utf-8") as fh:
            for rec in load_results(results_gz):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_path = args.out or work_jsonl
    summary = run_agent_bench(
        scenarios,
        provider,
        out_path=out_path,
        with_guard=with_guard,
        resume=args.resume,
        max_requests=args.max_requests,
        policy_path=args.policy,
        guard=guard,
        llm_provider=provider if with_guard else None,
        delay=args.delay,
    )

    # Compress working JSONL into results.jsonl.gz
    source = Path(out_path)
    if source.exists():
        mode_append = False
        with open_results_writer(run_dir, append=mode_append) as writer:
            with source.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    writer.write(json.loads(line))
        if source == work_jsonl and work_jsonl.exists():
            work_jsonl.unlink()

    metrics_dict = agent_summary_to_metrics(summary)
    metrics_dict["command"] = "agent-bench"
    metrics_dict["with_guard"] = with_guard
    config: Dict[str, Any] = {
        "command": "agent-bench",
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "with_guard": with_guard,
        "policy": str(args.policy) if args.policy else None,
        "s3_backend": args.s3_backend,
        "enable_s4": args.enable_s4,
        "max_requests": args.max_requests,
        "temperature": args.temperature,
        "seed": None if args.no_seed else args.seed,
        "resume": args.resume,
    }
    finalize_run_dir(
        run_dir,
        command="agent-bench",
        config=config,
        metrics=metrics_dict,
        argv=sys.argv[1:],
        provider=args.provider,
        model=args.model,
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    print(f"Run artefacts written to {run_dir}")
    print(json.dumps(summary, indent=2))
    print(
        f"provider_requests_used={summary.get('provider_requests', 0)}",
        file=sys.stderr,
    )
    return 0


def cmd_sentinel_proxy(args: argparse.Namespace) -> int:
    from llm_tester.sentinel.proxy import main as proxy_main

    argv: List[str] = []
    if args.policy:
        argv.extend(["--policy", str(args.policy)])
    if args.log:
        argv.extend(["--log", str(args.log)])
    if args.user_task:
        argv.extend(["--user-task", args.user_task])
    argv.extend(args.command)
    return proxy_main(argv)


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

    run_dir = prepare_run_dir(
        args,
        "run",
        name_parts=["run", args.dataset, args.provider, args.model or "default", args.guard],
    )
    results_gz = run_dir / "results.jsonl.gz"

    print(
        f"Dataset={args.dataset} prompts={len(selected)} "
        f"(indices {start}:{end}) provider={args.provider} "
        f"guard={args.guard} judge={args.judge}"
    )
    print(f"Run directory: {run_dir}")
    if args.resume:
        print(f"Resume enabled: skipping ids already in {results_gz}")
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

    # Dual-write: gzip artefacts + optional legacy --out JSONL.
    # run_evaluation writes plain JSONL; we use a work file then compress.
    work_jsonl = run_dir / "_results_work.jsonl"
    if args.resume and results_gz.exists():
        from llm_tester.artifacts.run_dir import load_results

        with work_jsonl.open("w", encoding="utf-8") as fh:
            for rec in load_results(results_gz):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif not args.resume and work_jsonl.exists():
        work_jsonl.unlink()

    summary = run_evaluation(
        selected,
        provider,
        judge=judge,
        guard=guard,
        out_path=work_jsonl,
        verbose=not args.quiet,
        start_number=start + 1,
        resume=args.resume,
        max_requests=args.max_requests,
        delay=args.delay,
    )

    if work_jsonl.exists():
        with open_results_writer(run_dir, append=False) as writer:
            with work_jsonl.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    writer.write(json.loads(line))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(work_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
        work_jsonl.unlink()

    metrics_dict = summary_to_metrics(
        summary,
        extra={
            "command": "run",
            "dataset": args.dataset,
            "provider": args.provider,
            "model": args.model or provider.info().get("model"),
            "guard": args.guard,
            "judge": args.judge,
        },
    )
    config: Dict[str, Any] = {
        "command": "run",
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "guard": args.guard,
        "judge": args.judge,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "range": args.range_spec,
        "temperature": args.temperature,
        "seed": None if args.no_seed else args.seed,
        "max_tokens": args.max_tokens,
        "max_requests": args.max_requests,
        "delay": args.delay,
        "max_retries": args.max_retries,
        "expand_conversations": args.expand_conversations,
        "resume": args.resume,
        "params": params.as_dict(),
    }
    finalize_run_dir(
        run_dir,
        command="run",
        config=config,
        metrics=metrics_dict,
        argv=sys.argv[1:],
        provider=args.provider,
        model=args.model or provider.info().get("model"),
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    print_summary(summary, results_gz)
    print(f"Run artefacts written to {run_dir}")
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


def cmd_index(args: argparse.Namespace) -> int:
    rows = build_index(args.runs_root)
    out = args.out or (Path(args.runs_root) / "index.csv")
    write_index_csv(rows, out)
    print(f"Indexed {len(rows)} run(s) → {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if len(args.runs) < 1:
        print("Provide at least one run directory.", file=sys.stderr)
        return 2
    comparison = compare_runs(args.runs)
    text = format_compare(
        comparison,
        fmt=args.fmt,
        table_number=args.table_number,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote comparison to {args.out}")
    else:
        print(text, end="" if text.endswith("\n") else "\n")
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from llm_tester.artifacts.run_dir import iter_run_dirs
    from llm_tester.figures.generate import generate_figures

    run_paths: List[Path] = []
    for p in args.runs:
        path = Path(p)
        if (path / "metrics.json").exists():
            run_paths.append(path)
        else:
            found = iter_run_dirs(path)
            if not found:
                print(f"No run artefacts under {path}", file=sys.stderr)
                return 1
            run_paths.extend(found)
    if not run_paths:
        print("No run directories to plot.", file=sys.stderr)
        return 1
    manifest = generate_figures(
        run_paths,
        args.out,
        lang=args.lang,
        figure_start=args.figure_start,
        table_start=args.table_start,
    )
    print(
        f"Wrote {len(manifest['figures'])} figure(s) and "
        f"{len(manifest['tables'])} table(s) to {args.out}"
    )
    print(f"Manifest: {manifest.get('manifest_path')}")
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
    if args.command == "guard-eval":
        return cmd_guard_eval(args)
    if args.command == "agent-bench":
        return cmd_agent_bench(args)
    if args.command == "sentinel-proxy":
        return cmd_sentinel_proxy(args)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "figures":
        return cmd_figures(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
