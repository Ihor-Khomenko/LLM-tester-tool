"""Console entry point: ``llmtest run``, ``report``, ``datasets``, ``guard-eval``, ``agent-bench``, ``index``, ``compare``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_tester.artifacts import (
    DEFAULT_RUNS_ROOT,
    RESULTS_PLAIN,
    collect_env,
    compare_metrics,
    ensure_plain_jsonl_for_resume,
    finalize_run,
    format_compare_markdown,
    format_compare_table,
    load_run,
    parse_extra_sha,
    resolve_run_dir,
    write_compare_csv,
    write_index,
)
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


def _add_artifact_args(parser: argparse.ArgumentParser) -> None:
    """Shared reproducible-run artifact flags."""
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Directory for run artifacts (config.yaml, env.json, results.jsonl.gz, "
            f"metrics.json, README.md). Default: {DEFAULT_RUNS_ROOT}/<date>_<name>"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name segment for the default --run-dir (runs/<date>_<name>)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"Root for auto run dirs (default: {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--extra-sha",
        action="append",
        default=[],
        metavar="NAME=SHA|PATH",
        help=(
            "Record an extra git SHA in env.json (repeatable). "
            "Forms: name=sha, name=/path/to/repo, or /path/to/repo "
            "(e.g. mcp-security-research)"
        ),
    )


def _parse_extra_shas(specs: Sequence[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for spec in specs or []:
        try:
            out.append(parse_extra_sha(spec))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise
    return out


def _namespace_to_config(args: argparse.Namespace, *, command: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Serialize CLI namespace to config.yaml payload (paths as strings; no secrets)."""
    cfg: Dict[str, Any] = {"command": command}
    skip = {"command"}
    for key, val in sorted(vars(args).items()):
        if key in skip:
            continue
        if isinstance(val, Path):
            cfg[key] = str(val)
        elif isinstance(val, list) and val and isinstance(val[0], Path):
            cfg[key] = [str(v) for v in val]
        else:
            cfg[key] = val
    if extra:
        cfg.update(extra)
    return cfg


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
            "JSONL output path during the run (default: <run-dir>/results.jsonl; "
            "finalized as results.jsonl.gz in the run directory)"
        ),
    )
    _add_artifact_args(parser)
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
    _add_artifact_args(ge)

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
        help=(
            "JSONL output path during the run (default: <run-dir>/results.jsonl; "
            "finalized as results.jsonl.gz)"
        ),
    )
    ab.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenario ids already present in --out",
    )
    ab.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Cap new provider calls this run",
    )
    ab.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries on HTTP 429 (default 5)",
    )
    ab.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )
    ab.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ab.add_argument("--no-seed", action="store_true")
    _add_artifact_args(ab)

    idx = sub.add_parser(
        "index",
        help="Build index.csv from a runs root (thesis experiments/runs/)",
    )
    idx.add_argument(
        "runs_root",
        type=Path,
        nargs="?",
        default=DEFAULT_RUNS_ROOT,
        help=f"Root directory containing run dirs (default: {DEFAULT_RUNS_ROOT})",
    )
    idx.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <runs-root>/index.csv)",
    )

    cmp = sub.add_parser(
        "compare",
        help="Diff metrics.json across two or more run directories",
    )
    cmp.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories to compare (at least two recommended)",
    )
    cmp.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional markdown table output path (thesis tables)",
    )
    cmp.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path (thesis tables)",
    )
    cmp.add_argument(
        "--json",
        action="store_true",
        help="Emit comparison rows as JSON",
    )

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
    return parser


def cmd_guard_eval(args: argparse.Namespace) -> int:
    from llm_tester.agent.eval import evaluate_traces, format_metrics
    from llm_tester.agent.provider_adapter import build_llm_chat
    from llm_tester.agent.traces import resolve_traces
    from llm_tester.sentinel.guard import build_sentinel
    from llm_tester.sentinel.policy import load_policy

    try:
        extra_shas = _parse_extra_shas(args.extra_sha)
    except ValueError:
        return 2

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

    run_dir = resolve_run_dir(
        args.run_dir,
        run_name=args.run_name,
        runs_root=args.runs_root,
        command="guard-eval",
        dataset=args.dataset,
        provider=args.provider,
        model=args.model,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = args.log
    if log_path is None:
        log_path = run_dir / "decisions.jsonl"

    guard = build_sentinel(policy=policy, llm_provider=llm, log_path=log_path)
    metrics = evaluate_traces(traces, guard, collect_records=True)

    config = _namespace_to_config(
        args,
        command="guard-eval",
        extra={"thresholds": policy.to_dict(), "run_dir": str(run_dir)},
    )
    env = collect_env(
        command="guard-eval",
        provider=args.provider,
        model=args.model,
        extra_shas=extra_shas,
    )
    art = finalize_run(
        run_dir,
        config=config,
        env=env,
        metrics=metrics.to_dict(),
        results_records=metrics.records,
    )
    print(f"Run artifacts written to {art.run_dir}")

    if args.json:
        print(json.dumps(metrics.to_dict(), indent=2))
    else:
        abl = []
        if args.no_s1:
            abl.append("-S1")
        if args.no_s2:
            abl.append("-S2")
        if args.no_s3:
            abl.append("-S3")
        if args.only_s4:
            abl.append("only-S4")
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
    from llm_tester.agent.agent_bench import DEFAULT_SCENARIOS, run_agent_bench
    from llm_tester.agent.provider_adapter import build_llm_chat
    from llm_tester.sentinel.guard import build_sentinel
    from llm_tester.sentinel.policy import load_policy

    try:
        extra_shas = _parse_extra_shas(args.extra_sha)
    except ValueError:
        return 2

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
    policy_dict: Dict[str, Any] = {}
    if with_guard:
        policy = load_policy(args.policy)
        if args.s3_backend:
            policy.s3_backend = args.s3_backend
        if args.enable_s4:
            policy.enable_s4 = True
        policy_dict = policy.to_dict()
        if args.enable_s4 or args.s3_backend:
            guard = build_sentinel(policy=policy, llm_provider=provider)

    run_dir = resolve_run_dir(
        args.run_dir,
        run_name=args.run_name,
        runs_root=args.runs_root,
        command="agent-bench",
        provider=args.provider,
        model=args.model,
        dataset="with-guard" if with_guard else "without-guard",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (run_dir / RESULTS_PLAIN)
    if args.resume:
        out_path = ensure_plain_jsonl_for_resume(run_dir, out_path)

    summary = run_agent_bench(
        DEFAULT_SCENARIOS,
        provider,
        out_path=out_path,
        with_guard=with_guard,
        resume=args.resume,
        max_requests=args.max_requests,
        policy_path=args.policy,
        guard=guard,
        llm_provider=provider if with_guard else None,
    )
    # Derive rates for metrics.json
    evaluated = max(0, summary.get("total", 0) - summary.get("errors", 0))
    metrics = {
        **summary,
        "evaluated": evaluated,
        "attack_success_rate": (
            (summary["attack_successes"] / evaluated) if evaluated else None
        ),
        "block_rate": (summary["blocked"] / evaluated) if evaluated else None,
        "guard_enabled": with_guard,
    }
    config = _namespace_to_config(
        args,
        command="agent-bench",
        extra={
            "thresholds": policy_dict,
            "run_dir": str(run_dir),
            "with_guard": with_guard,
        },
    )
    env = collect_env(
        command="agent-bench",
        provider=args.provider,
        model=args.model,
        extra_shas=extra_shas,
    )
    art = finalize_run(
        run_dir,
        config=config,
        env=env,
        metrics=metrics,
        results_jsonl=out_path,
        remove_plain_results=out_path.parent == run_dir,
    )
    print(f"Run artifacts written to {art.run_dir}")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    root = Path(args.runs_root)
    if not root.exists():
        print(f"Runs root not found: {root}", file=sys.stderr)
        return 1
    out = write_index(root, out=args.out)
    print(f"Wrote {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    if len(args.runs) < 1:
        print("Provide at least one run directory.", file=sys.stderr)
        return 2
    arts = []
    for p in args.runs:
        try:
            arts.append(load_run(p))
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Failed to load run {p}: {exc}", file=sys.stderr)
            return 1
    rows = compare_metrics(arts)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(format_compare_table(arts, rows))
    if args.markdown:
        Path(args.markdown).write_text(
            format_compare_markdown(arts, rows) + "\n", encoding="utf-8"
        )
        print(f"Wrote markdown table to {args.markdown}", file=sys.stderr)
    if args.csv:
        write_compare_csv(args.csv, arts, rows)
        print(f"Wrote CSV table to {args.csv}", file=sys.stderr)
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
    try:
        extra_shas = _parse_extra_shas(args.extra_sha)
    except ValueError:
        return 2

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

    run_dir = resolve_run_dir(
        args.run_dir,
        run_name=args.run_name,
        runs_root=args.runs_root,
        command="run",
        dataset=args.dataset,
        provider=args.provider,
        model=args.model,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (run_dir / RESULTS_PLAIN)
    if args.resume:
        out_path = ensure_plain_jsonl_for_resume(run_dir, out_path)

    print(
        f"Dataset={args.dataset} prompts={len(selected)} "
        f"(indices {start}:{end}) provider={args.provider} "
        f"guard={args.guard} judge={args.judge}"
    )
    print(f"Run directory: {run_dir}")
    if args.resume:
        print(f"Resume enabled: skipping ids already in {out_path}")
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
    j_provider_name = None
    j_model = None
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
        out_path=out_path,
        verbose=not args.quiet,
        start_number=start + 1,
        resume=args.resume,
        max_requests=args.max_requests,
        delay=args.delay,
    )
    info = provider.info()
    metrics = {
        "total": summary.total,
        "evaluated": summary.evaluated,
        "skipped": summary.skipped,
        "resumed_skipped": summary.resumed_skipped,
        "errors": summary.errors,
        "refusals": summary.refusals,
        "attack_successes": summary.attack_successes,
        "blocked": summary.blocked,
        "requests": summary.requests,
        "stopped_max_requests": summary.stopped_max_requests,
        "refusal_rate": summary.refusal_rate,
        "attack_success_rate": summary.attack_success_rate,
        "mean_latency_s": summary.mean_latency_s,
    }
    config = _namespace_to_config(
        args,
        command="run",
        extra={
            "run_dir": str(run_dir),
            "params": params.as_dict(),
            "resolved_model": info.get("model"),
            "resolved_base_url": info.get("base_url"),
            "range_start": start,
            "range_end": end,
            "n_prompts": len(selected),
        },
    )
    env = collect_env(
        command="run",
        provider=args.provider,
        model=info.get("model") or args.model,
        judge_provider=j_provider_name,
        judge_model=j_model,
        extra_shas=extra_shas,
    )
    art = finalize_run(
        run_dir,
        config=config,
        env=env,
        metrics=metrics,
        results_jsonl=out_path,
        remove_plain_results=out_path.parent == run_dir,
    )
    print_summary(summary, art.results_path or out_path)
    print(f"Run artifacts written to {art.run_dir}")
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
    if args.command == "guard-eval":
        return cmd_guard_eval(args)
    if args.command == "agent-bench":
        return cmd_agent_bench(args)
    if args.command == "index":
        return cmd_index(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "sentinel-proxy":
        return cmd_sentinel_proxy(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
