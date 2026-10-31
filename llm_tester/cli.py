"""Console entry point: ``llmtest run``, ``report``, ``datasets``, ``guard-eval``,
``baselines-eval``, ``agentdojo``, ``ablation``, ``guard-calibrate``,
``grey-zone-rebuild``, ``import-mcp-sim``, ``adaptive-attack``,
``mcp-scenario-eval``, ``agent-bench``, ``index``, ``compare``, ``figures``.
"""

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
        help="Enable S4 LLM judge (grey zone); needs --provider or --s4-mock",
    )
    ge.add_argument(
        "--s4-mock",
        action="store_true",
        help=(
            "Use offline MockS4Chat judge for the S4 path (CI; no network). "
            "Implies enable_s4=True."
        ),
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
    ge.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="Ollama options.num_thread (default 1 when --seed is set)",
    )
    ge.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat evaluate_traces K times; report mean/std of key metrics",
    )
    ge.add_argument(
        "--repeat-seed-stride",
        type=int,
        default=0,
        help="With --repeats K>1, use seed, seed+stride, … per repeat (0=same seed)",
    )
    add_run_dir_args(ge, default_name_help="guard-eval_<dataset>")

    be = sub.add_parser(
        "baselines-eval",
        help=(
            "Evaluate pluggable baselines (spotlighting / prompt_guard2 / "
            "progent / task_shield / sentinel) on the same traces as guard-eval"
        ),
    )
    be.add_argument(
        "--baseline",
        required=True,
        choices=[
            "spotlighting",
            "prompt_guard2",
            "progent",
            "task_shield",
            "sentinel",
        ],
        help="Defense baseline to evaluate",
    )
    be.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "injecagent", "jsonl"],
    )
    be.add_argument("--dataset-path", type=Path, default=None)
    be.add_argument(
        "--download-injecagent",
        action="store_true",
        help="Download InjecAgent (needs network; not for CI)",
    )
    be.add_argument("--limit", type=int, default=None)
    be.add_argument(
        "--spotlight-mode",
        default="datamark",
        choices=["datamark", "encode"],
        help="Spotlighting transform (Hines et al.)",
    )
    be.add_argument(
        "--prompt-guard-model",
        default=None,
        help="HF id or local path for PromptGuard 2 (default Llama-Prompt-Guard-2-22M)",
    )
    be.add_argument(
        "--prompt-guard-threshold",
        type=float,
        default=0.5,
        help="Injection score threshold for PromptGuard 2",
    )
    be.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS.keys()),
        help="LLM for task_shield / sentinel S3-llm|S4 (ollama recommended)",
    )
    be.add_argument("--model", default=None, help="Model id (e.g. qwen2.5:3b)")
    be.add_argument("--base-url", default=None)
    be.add_argument("--policy", type=Path, default=None, help="Sentinel policy (baseline=sentinel)")
    be.add_argument("--json", action="store_true")
    be.add_argument("--max-retries", type=int, default=5)
    be.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    be.add_argument("--seed", type=int, default=DEFAULT_SEED)
    be.add_argument("--no-seed", action="store_true")
    be.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="Ollama options.num_thread (default 1 when --seed is set); recorded in manifest",
    )
    add_run_dir_args(be, default_name_help="baselines-eval_<baseline>_<dataset>")

    adj = sub.add_parser(
        "agentdojo",
        help=(
            "AgentDojo-style suite with MCP-Sentinel/baselines as defenses. "
            "Default --suite mock is offline; --real needs upstream AgentDojo + Ollama"
        ),
    )
    adj.add_argument(
        "--suite",
        default="mock",
        help="mock (CI default) | upstream suite name with --real (e.g. workspace)",
    )
    adj.add_argument(
        "--defense",
        default="sentinel",
        choices=[
            "none",
            "sentinel",
            "spotlighting",
            "prompt_guard2",
            "progent",
            "task_shield",
        ],
        help="Defense plugged into the suite",
    )
    adj.add_argument(
        "--real",
        action="store_true",
        help=(
            "Run upstream AgentDojo 0.1.35+ agent pipeline + env "
            "(requires package). Use --provider mock for CI."
        ),
    )
    adj.add_argument(
        "--benchmark-version",
        default="v1.2.2",
        help="AgentDojo benchmark version for --real (default v1.2.2)",
    )
    adj.add_argument(
        "--attack",
        default="important_instructions",
        help="AgentDojo attack name for --real injection jobs "
        "(default important_instructions; 'none' skips injections)",
    )
    adj.add_argument("--limit", type=int, default=None)
    adj.add_argument("--policy", type=Path, default=None)
    adj.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS.keys()) + ["mock"],
        help=(
            "Agent LLM for --real: mock (CI default), ollama, openrouter, … "
            "Also used for task_shield defense"
        ),
    )
    adj.add_argument("--model", default=None, help="e.g. qwen2.5:3b")
    adj.add_argument("--base-url", default=None)
    adj.add_argument("--json", action="store_true")
    adj.add_argument("--max-retries", type=int, default=5)
    adj.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    adj.add_argument("--seed", type=int, default=DEFAULT_SEED)
    adj.add_argument("--no-seed", action="store_true")
    adj.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="Ollama options.num_thread (default 1 when --seed is set); recorded in manifest",
    )
    add_run_dir_args(adj, default_name_help="agentdojo_<suite>_<defense>")

    abl = sub.add_parser(
        "ablation",
        help=(
            "Run stage/aggregation/checkpoint ablation matrix "
            "(S1‥full × noisy_or|max|weighted × ±T0/T1); offline"
        ),
    )
    abl.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "injecagent", "jsonl", "adaptive"],
        help="synthetic | injecagent | jsonl | adaptive (S2-evasion suite)",
    )
    abl.add_argument("--dataset-path", type=Path, default=None)
    abl.add_argument("--limit", type=int, default=None)
    abl.add_argument("--policy", type=Path, default=None)
    abl.add_argument(
        "--aggregations",
        default="noisy_or,max,weighted",
        help="Comma list: noisy_or,max,weighted",
    )
    abl.add_argument(
        "--no-checkpoint-ablation",
        action="store_true",
        help="Skip ±T0/T1 variants (stage ladder only)",
    )
    abl.add_argument(
        "--write-adaptive",
        action="store_true",
        help="Regenerate data/agent/adaptive_s2_traces.jsonl from synthetic seeds",
    )
    abl.add_argument("--json", action="store_true")
    add_run_dir_args(abl, default_name_help="ablation_<dataset>")

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

    gc = sub.add_parser(
        "guard-calibrate",
        help="Tune tau_low/tau_high on a train split; evaluate on held-out",
    )
    gc.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "injecagent", "jsonl"],
        help="Dataset to split (injecagent recommended for calibration)",
    )
    gc.add_argument("--dataset-path", type=Path, default=None)
    gc.add_argument(
        "--download-injecagent",
        action="store_true",
        help="Download InjecAgent JSON into .cache/ (needs network)",
    )
    gc.add_argument("--limit", type=int, default=None)
    gc.add_argument("--policy", type=Path, default=None)
    gc.add_argument(
        "--split-by",
        default="tool",
        choices=["tool", "attack_type", "id_hash"],
        help="Hold out whole groups (attacker tool / attack type / id hash)",
    )
    gc.add_argument(
        "--train-frac",
        type=float,
        default=0.5,
        help="Fraction of groups used for threshold search",
    )
    gc.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed (or sole seed when --seeds/--n-seeds unset)",
    )
    gc.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for repeated calibration (e.g. 0,1,2,3,4)",
    )
    gc.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Run N calibrations with seeds seed..seed+N-1; report mean±95%% CI",
    )
    gc.add_argument(
        "--objective",
        default="youden",
        choices=["youden", "f1", "block_f1", "tpr", "f1_fpr"],
        help="Train objective for grid search",
    )
    gc.add_argument(
        "--max-fpr",
        type=float,
        default=0.15,
        help="Discard train candidates with any-intervention FPR above this",
    )
    gc.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON mirror of calibration result (also written under --run-dir)",
    )
    gc.add_argument("--json", action="store_true", help="Print full result JSON only on stdout")
    add_run_dir_args(gc, default_name_help="guard-calibrate_<dataset>")

    gzr = sub.add_parser(
        "grey-zone-rebuild",
        help=(
            "Rebuild an S4 grey-zone JSONL subset from a prior guard-eval "
            "run's results (escalated / S4 stage filter)"
        ),
    )
    gzr.add_argument(
        "--from-run",
        type=Path,
        required=True,
        help="Prior guard-eval run directory (with results.jsonl.gz)",
    )
    gzr.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "injecagent", "jsonl"],
        help="Same dataset used for the source run",
    )
    gzr.add_argument("--dataset-path", type=Path, default=None)
    gzr.add_argument(
        "--download-injecagent",
        action="store_true",
        help="Download InjecAgent JSON if needed",
    )
    gzr.add_argument("--limit", type=int, default=None)
    gzr.add_argument(
        "--filter",
        default="escalated",
        choices=["escalated", "s4-stage", "both"],
        help="Which result rows count as grey-zone (default: escalated)",
    )
    gzr.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output mcp-sentinel-trace JSONL path",
    )
    gzr.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero when the filter matches no S4/escalated decisions "
            "(default: warn loudly on stderr but still write an empty JSONL)"
        ),
    )
    gzr.add_argument("--json", action="store_true", help="Print manifest JSON")

    ims = sub.add_parser(
        "import-mcp-sim",
        help=(
            "Convert mcp-security-research scenario export (schema 1.0) into "
            "mcp-sentinel-trace/v1 JSONL for guard-eval (T0/T1/T2 mapping)"
        ),
    )
    ims.add_argument(
        "--input",
        type=Path,
        default=None,
        help="MSR schema 1.0 JSONL (from python -m scenarios.export)",
    )
    ims.add_argument(
        "--research-root",
        type=Path,
        default=None,
        help=(
            "Local mcp-security-research checkout; runs python -m scenarios.export "
            "then converts (overrides --input)"
        ),
    )
    ims.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output mcp-sentinel-trace/v1 JSONL path",
    )
    ims.add_argument(
        "--id",
        action="append",
        dest="scenario_ids",
        default=None,
        help="Restrict to scenario id(s) (repeatable)",
    )
    ims.add_argument("--lang", choices=["en", "uk"], default="en")
    ims.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional import manifest path (default: <out>_import_manifest.json)",
    )
    ims.add_argument("--json", action="store_true", help="Print import manifest JSON")

    aa = sub.add_parser(
        "adaptive-attack",
        help=(
            "Sentinel-aware adaptive attack harness (A-para/A-split/A-judge/A-iter). "
            "CI uses --mock (scripted defense); live AgentDojo wiring is library-level. "
            "Counts LLM calls via an optional request ledger / --run-cap."
        ),
    )
    aa.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for progress.json / results.jsonl / summary.json",
    )
    aa.add_argument(
        "--defense",
        action="append",
        dest="defenses",
        default=None,
        help="Defense id (repeatable). Default: none,sentinel",
    )
    aa.add_argument(
        "--family",
        action="append",
        dest="families",
        default=None,
        help=(
            "Attack family (repeatable): important_instructions|a_para|a_split|"
            "a_judge|a_iter. Default: a_para,a_iter"
        ),
    )
    aa.add_argument(
        "--k",
        type=int,
        default=3,
        help="Max adaptive tries for a_iter (default 3; use 5 for full budget)",
    )
    aa.add_argument("--seed", type=int, default=42)
    aa.add_argument("--limit", type=int, default=None, help="Cap seed jobs")
    aa.add_argument(
        "--run-cap",
        type=int,
        default=None,
        help="Hard LLM-request cap for this process (budget gate)",
    )
    aa.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Optional JSON ledger path (default: runs/<date>_openrouter_ledger.json)",
    )
    aa.add_argument(
        "--daily-cap",
        type=int,
        default=None,
        help="Override OR_DAILY_CAP for the ledger (default 45)",
    )
    aa.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Scripted offline defense (default; required for CI)",
    )
    aa.add_argument(
        "--no-mock",
        action="store_true",
        help="Refuse to run without a custom library run_try (live path)",
    )
    aa.add_argument("--json", action="store_true", help="Print summary JSON")

    mse = sub.add_parser(
        "mcp-scenario-eval",
        help=(
            "Evaluate mcp-security-research scenarios with and without "
            "MCP-Sentinel (ASR, utility proxies, detections). Offline by default "
            "on the bundled fixture; optional --research-root / --input."
        ),
    )
    mse.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for progress.json / results.jsonl / summary.json",
    )
    mse.add_argument(
        "--input",
        type=Path,
        default=None,
        help="MSR schema 1.0 or v1 JSONL (default: bundled mcp_sim_import_fixture)",
    )
    mse.add_argument(
        "--research-root",
        type=Path,
        default=None,
        help="Local mcp-security-research checkout (runs scenarios.export)",
    )
    mse.add_argument(
        "--defense",
        action="append",
        dest="defenses",
        default=None,
        help="Defense id (repeatable). Default: none,sentinel",
    )
    mse.add_argument(
        "--id",
        action="append",
        dest="scenario_ids",
        default=None,
        help="Restrict to scenario id(s) (repeatable)",
    )
    mse.add_argument(
        "--lang",
        choices=["en", "uk"],
        default="en",
        help="Scenario language; user + payloads are aligned to this (default en)",
    )
    mse.add_argument(
        "--no-align-language",
        action="store_true",
        help="Disable user/payload language alignment",
    )
    mse.add_argument("--limit", type=int, default=None)
    mse.add_argument("--policy", type=Path, default=None)
    mse.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
        help=(
            "replay=detection-on-recorded-traces (TPR/FPR, not ASR); "
            "live=LLM agent + mock tools (measured ASR). Default: replay"
        ),
    )
    mse.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Expand each scenario into N paraphrased/retargeted variants (default 1)",
    )
    mse.add_argument(
        "--min-n-for-ci",
        type=int,
        default=5,
        help="Refuse Wilson CI endpoints when n < this (default 5)",
    )
    mse.add_argument(
        "--run-cap",
        type=int,
        default=None,
        help="Hard LLM-request cap for live mode (same idea as adaptive-attack)",
    )
    mse.add_argument(
        "--resume",
        action="store_true",
        help="Live mode: continue from out-dir/progress.json + results.jsonl",
    )
    mse.add_argument(
        "--order",
        choices=["fixed", "alternate"],
        default="fixed",
        help=(
            "Live case order: fixed=load order; alternate=interleave attack/benign "
            "(frozen into progress.json for resume)"
        ),
    )
    mse.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help="Optional JSON ledger path for live-mode agent requests",
    )
    mse.add_argument(
        "--daily-cap",
        type=int,
        default=None,
        help="Override OR_DAILY_CAP for the ledger (default 45)",
    )
    mse.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS.keys()),
        help="Live-mode agent provider (openrouter recommended); omit with --mock",
    )
    mse.add_argument("--model", default=None, help="Live-mode agent model id")
    mse.add_argument("--base-url", default=None, help="Override provider base URL")
    mse.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Live-mode ReAct turns per scenario (default 4; ≤4 agent requests)",
    )
    mse.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Plan cases and print request upper bounds without LLM calls or "
            "guard execution (works with --mode live|replay)"
        ),
    )
    mse.add_argument(
        "--enable-s4",
        action="store_true",
        help="Enable S4 on sentinel (needs --s4-mock offline, or a live provider)",
    )
    mse.add_argument(
        "--s4-mock",
        action="store_true",
        help="Offline MockS4Chat when --enable-s4",
    )
    mse.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Offline mocks (default; required for CI)",
    )
    mse.add_argument(
        "--no-mock",
        action="store_true",
        help="Use a real --provider for live mode (reads OPENROUTER_API_KEY from env)",
    )
    mse.add_argument(
        "--proxy-mode",
        action="store_true",
        help=(
            "Mark summary as T0/T1/T2-comparable to sentinel-proxy "
            "(still offline trace replay unless a live proxy is attached)"
        ),
    )
    mse.add_argument("--seed", type=int, default=42)
    mse.add_argument("--json", action="store_true", help="Print summary JSON")

    sp = sub.add_parser(
        "sentinel-proxy",
        help=(
            "Transparent MCP proxy (stdio JSON-RPC; optional HTTP) applying "
            "Sentinel T0/T1/T2 with block/sanitize/allow + JSONL audit"
        ),
    )
    sp.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Proxy YAML (threshold default 0.7; enable_s4; audit_log; transport)",
    )
    sp.add_argument("--policy", type=Path, default=None, help="Full Sentinel policy YAML")
    sp.add_argument("--log", type=Path, default=None, help="Decision / audit JSONL path")
    sp.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="JSONL audit log (T0/T1/T2 actions + latency)",
    )
    sp.add_argument("--user-task", default="", help="Trusted user task for provenance")
    sp.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Block threshold tau_high (default 0.7)",
    )
    sp.add_argument(
        "--enable-s4",
        action="store_true",
        help="Optional S4 judge (needs --provider; off by default)",
    )
    sp.add_argument(
        "--provider",
        default=None,
        choices=sorted(PROVIDERS.keys()),
        help="LLM provider for optional S4",
    )
    sp.add_argument("--model", default=None, help="S4 model id")
    sp.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="stdio (default) or http JSON-RPC front-end",
    )
    sp.add_argument(
        "--listen",
        default=None,
        help="host:port for --transport http (default 127.0.0.1:8765)",
    )
    sp.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Backend MCP server command after --",
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
    from llm_tester.agent.repeats import aggregate_repeat_metrics, format_repeat_aggregate
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

    use_s4_mock = bool(getattr(args, "s4_mock", False))
    want_llm = (
        args.only_s4
        or args.enable_s4
        or use_s4_mock
        or (args.s3_backend == "llm")
        or (policy.s3_backend == "llm")
    )
    if args.only_s4 or args.enable_s4 or use_s4_mock:
        policy.enable_s4 = True
    elif not args.only_s4:
        # Default offline path: no S4 unless explicitly enabled.
        if not args.enable_s4 and not use_s4_mock:
            policy.enable_s4 = False

    if want_llm and not args.provider and not use_s4_mock:
        print(
            "Warning: S3/S4 LLM backend requested but --provider not set; "
            "falling back to lite S3 / no S4. Pass --s4-mock for offline S4.",
            file=sys.stderr,
        )
        if policy.s3_backend == "llm":
            policy.s3_backend = "lite"
        policy.enable_s4 = False
        want_llm = False

    abl = []
    if args.no_s1:
        abl.append("-S1")
    if args.no_s2:
        abl.append("-S2")
    if args.no_s3:
        abl.append("-S3")
    if args.only_s4:
        abl.append("only-S4")
    if use_s4_mock:
        abl.append("s4-mock")
    abl_slug = "".join(abl) if abl else "full"

    repeats = max(1, int(getattr(args, "repeats", 1) or 1))
    base_seed = None if args.no_seed else args.seed
    stride = int(getattr(args, "repeat_seed_stride", 0) or 0)

    run_dir = prepare_run_dir(
        args,
        "guard-eval",
        name_parts=["guard-eval", args.dataset, abl_slug],
    )

    per_repeat_metrics: List[Dict[str, Any]] = []
    last_metrics = None
    last_records: List[Dict[str, Any]] = []

    for i in range(repeats):
        seed_i = base_seed
        if base_seed is not None and stride:
            seed_i = int(base_seed) + i * stride
        llm = None
        if use_s4_mock:
            from llm_tester.agent.s4_mock import MockS4Chat

            llm = MockS4Chat(seed=seed_i if seed_i is not None else 42)
        elif want_llm and args.provider:
            llm = build_llm_chat(
                args.provider,
                args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                seed=seed_i,
                max_retries=args.max_retries,
                num_thread=getattr(args, "num_thread", None),
            )
        log_path = args.log
        if log_path is None:
            log_path = (
                run_dir / f"sentinel_decisions_r{i}.jsonl.gz"
                if repeats > 1
                else run_dir / "sentinel_decisions.jsonl.gz"
            )
        guard = build_sentinel(policy=policy, llm_provider=llm, log_path=log_path)
        try:
            metrics, records = evaluate_traces(traces, guard, collect_records=True)
        finally:
            if getattr(guard, "logger", None) is not None:
                guard.logger.close()
        mdict = metrics.to_dict()
        mdict["repeat_index"] = i
        mdict["seed"] = seed_i
        per_repeat_metrics.append(mdict)
        last_metrics = metrics
        last_records = records
        if repeats > 1:
            with open_results_writer(
                run_dir / f"repeat_{i}", append=False
            ) as writer:
                for rec in records:
                    writer.write(rec)

    assert last_metrics is not None
    with open_results_writer(run_dir, append=False) as writer:
        for rec in last_records:
            writer.write(rec)

    metrics_dict = last_metrics.to_dict()
    metrics_dict["dataset"] = args.dataset
    metrics_dict["command"] = "guard-eval"
    metrics_dict["repeats"] = repeats
    if repeats > 1:
        agg = aggregate_repeat_metrics(per_repeat_metrics)
        metrics_dict["per_repeat"] = per_repeat_metrics
        metrics_dict["aggregate"] = agg

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
        "s4_mock": use_s4_mock,
        "s3_backend": policy.s3_backend,
        "provider": "mock-s4" if use_s4_mock else args.provider,
        "model": "mock-s4" if use_s4_mock else args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "seed": base_seed,
        "num_thread": getattr(args, "num_thread", None),
        "repeats": repeats,
        "repeat_seed_stride": stride,
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
        provider=("mock-s4" if use_s4_mock else args.provider),
        model=("mock-s4" if use_s4_mock else args.model),
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    print(f"Run artefacts written to {run_dir}")

    if args.json:
        print(json.dumps(metrics_dict, indent=2))
    else:
        title = f"guard-eval dataset={args.dataset}" + (
            f" ablation={','.join(abl)}" if abl else ""
        )
        print(format_metrics(last_metrics, title=title))
        if repeats > 1:
            print()
            print(
                format_repeat_aggregate(
                    metrics_dict["aggregate"],
                    repeats=repeats,
                    title=title + " aggregate",
                )
            )
        print(
            "\nNote: numbers are measured on this run's traces (preliminary). "
            "Not claimed as published results."
        )
    return 0


def cmd_grey_zone_rebuild(args: argparse.Namespace) -> int:
    from llm_tester.agent.grey_zone import EmptyGreyZoneError, rebuild_grey_zone_subset
    from llm_tester.agent.traces import TraceSchemaError, resolve_traces

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

    try:
        manifest = rebuild_grey_zone_subset(
            args.from_run,
            traces,
            out_path=args.out,
            filter_mode=args.filter,
            strict=bool(getattr(args, "strict", False)),
        )
    except EmptyGreyZoneError as exc:
        print(f"grey-zone-rebuild failed (strict): {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"grey-zone-rebuild failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(
            f"Wrote {manifest['n']} grey-zone traces to {manifest['out_path']} "
            f"(filter={manifest['filter']}; source={manifest['source_run']})"
        )
        print(f"Manifest: {manifest.get('manifest_path')}")
        if manifest.get("empty"):
            print(
                "Note: empty grey-zone subset — see WARNING on stderr.",
                file=sys.stderr,
            )
    return 0


def cmd_import_mcp_sim(args: argparse.Namespace) -> int:
    from llm_tester.agent.import_mcp_sim import import_mcp_sim

    if args.research_root is None and args.input is None:
        print(
            "import-mcp-sim requires --input JSONL or --research-root "
            "(mcp-security-research checkout).",
            file=sys.stderr,
        )
        return 2
    try:
        manifest = import_mcp_sim(
            input_path=args.input,
            research_root=args.research_root,
            out_path=args.out,
            lang=args.lang,
            scenario_ids=args.scenario_ids,
            manifest_path=args.manifest,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"import-mcp-sim failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(
            f"Wrote {manifest['n_written']} v1 traces to {manifest['out_path']} "
            f"(attack={manifest['n_attack']}, benign={manifest['n_benign']}; "
            f"T0={manifest['by_checkpoint']['T0']}, "
            f"T1={manifest['by_checkpoint']['T1']}, "
            f"T2={manifest['by_checkpoint']['T2']})"
        )
        print(f"Manifest: {manifest.get('manifest_path')}")
    return 0 if manifest.get("n_written", 0) > 0 else 3


def cmd_adaptive_attack(args: argparse.Namespace) -> int:
    from llm_tester.agent.adaptive_eval import AdaptiveEvalConfig, run_adaptive_eval
    from llm_tester.budget.ledger import RequestLedger

    if args.no_mock:
        print(
            "Live adaptive-attack AgentDojo wiring is research-box specific. "
            "Use --mock for CI, or call llm_tester.attacks.adaptive.run_adaptive_job "
            "with a custom run_try.",
            file=sys.stderr,
        )
        return 2

    ledger = None
    if args.ledger is not None or args.daily_cap is not None or args.run_cap is not None:
        kwargs: Dict[str, Any] = {}
        if args.ledger is not None:
            kwargs["path"] = Path(args.ledger)
        if args.daily_cap is not None:
            kwargs["daily_cap"] = int(args.daily_cap)
        ledger = RequestLedger(**kwargs)

    cfg = AdaptiveEvalConfig(
        defenses=list(args.defenses) if args.defenses else ["none", "sentinel"],
        families=list(args.families) if args.families else ["a_para", "a_iter"],
        k=int(args.k),
        seed=int(args.seed),
        limit=args.limit,
        run_cap=args.run_cap,
        mock=True,
    )
    try:
        summary = run_adaptive_eval(Path(args.out_dir), config=cfg, ledger=ledger)
    except Exception as exc:  # noqa: BLE001
        print(f"adaptive-attack failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"adaptive-attack: {summary.get('n_results', 0)} results → "
            f"{summary.get('results_path')}"
        )
        by = summary.get("by_defense") or {}
        for name, block in by.items():
            asr1 = (block.get("ASR@1") or {}).get("p")
            asrk = (block.get("ASR@k") or {}).get("p")
            print(f"  {name}: ASR@1={asr1}  ASR@k={asrk}  n={block.get('n')}")
        est = summary.get("request_estimates") or {}
        if est:
            print("Request upper bounds (per job, early-stop may use fewer):")
            for fam, sides in est.items():
                ws = sides.get("with_sentinel") or {}
                print(
                    f"  {fam}+sentinel ≤ {ws.get('total_upper_bound')} "
                    f"(agent={ws.get('agent')}, defense={ws.get('defense')}, "
                    f"attacker={ws.get('attacker')})"
                )
    return 0


def cmd_mcp_scenario_eval(args: argparse.Namespace) -> int:
    from llm_tester.agent.mcp_scenario_eval import (
        McpScenarioEvalConfig,
        run_mcp_scenario_eval,
    )
    from llm_tester.budget.ledger import RequestLedger

    use_mock = not bool(getattr(args, "no_mock", False))
    mode = str(args.mode or "replay")
    if mode == "live" and not use_mock and not args.provider:
        print(
            "live --no-mock requires --provider (e.g. openrouter). "
            "API keys are read from the environment only.",
            file=sys.stderr,
        )
        return 2

    ledger = None
    if args.ledger is not None or args.daily_cap is not None or args.run_cap is not None:
        kwargs: Dict[str, Any] = {}
        if args.ledger is not None:
            kwargs["path"] = Path(args.ledger)
        if args.daily_cap is not None:
            kwargs["daily_cap"] = int(args.daily_cap)
        ledger = RequestLedger(**kwargs)

    provider = None
    if mode == "live" and not use_mock and args.provider:
        from llm_tester.agent.provider_adapter import build_llm_chat

        provider = build_llm_chat(
            args.provider,
            args.model,
            base_url=args.base_url,
            temperature=DEFAULT_TEMPERATURE,
            seed=args.seed,
        )

    cfg = McpScenarioEvalConfig(
        defenses=list(args.defenses) if args.defenses else ["none", "sentinel"],
        mock=use_mock,
        enable_s4=bool(args.enable_s4),
        s4_mock=bool(args.s4_mock),
        no_s4=not (args.enable_s4 or args.s4_mock),
        limit=args.limit,
        seed=int(args.seed),
        proxy_mode=bool(args.proxy_mode),
        mode=mode,
        n_variants=max(1, int(args.variants or 1)),
        min_n_for_ci=max(1, int(args.min_n_for_ci or 5)),
        align_language=not bool(args.no_align_language),
        run_cap=args.run_cap,
        max_turns=max(1, int(args.max_turns or 4)),
        dry_run=bool(getattr(args, "dry_run", False)),
        resume=bool(getattr(args, "resume", False)),
        order=str(getattr(args, "order", None) or "fixed"),
    )
    try:
        summary = run_mcp_scenario_eval(
            Path(args.out_dir),
            config=cfg,
            input_path=args.input,
            research_root=args.research_root,
            policy_path=args.policy,
            scenario_ids=args.scenario_ids,
            lang=args.lang,
            ledger=ledger,
            provider=provider,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"mcp-scenario-eval failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif summary.get("dry_run"):
        tot = summary.get("request_upper_bound_total") or {}
        print(
            f"mcp-scenario-eval dry-run ({summary.get('mode')}): "
            f"{summary.get('n_cases', 0)} cases × "
            f"{summary.get('n_defenses', 0)} defenses"
        )
        print(
            f"  request upper bound total ≤ {tot.get('total_upper_bound')} "
            f"(agent={tot.get('agent')}, defense={tot.get('defense')})"
        )
        for case in (summary.get("planned_cases") or [])[:20]:
            print(
                f"  - {case.get('id')} [{case.get('label')}/"
                f"{case.get('scenario_family')}]"
            )
        if len(summary.get("planned_cases") or []) > 20:
            print(f"  … ({len(summary['planned_cases']) - 20} more)")
    else:
        print(
            f"mcp-scenario-eval ({summary.get('config', {}).get('mode')}): "
            f"{summary.get('n_scenarios', 0)} scenarios → "
            f"{summary.get('results_path')}"
        )
        for name, block in (summary.get("by_defense") or {}).items():
            util = (block.get("utility") or {}).get("utility_benign")
            if block.get("metric_mode") == "live_agent":
                asr = (block.get("asr") or {}).get("asr")
                print(f"  {name}: ASR={asr}  utility_benign={util}")
            else:
                det = block.get("detection") or block.get("detections") or {}
                print(
                    f"  {name}: TPR={det.get('tpr')}  FPR={det.get('fpr')}  "
                    f"utility_benign={util}"
                )
        est = summary.get("request_estimates") or {}
        if est:
            print("Request upper bounds (per scenario):")
            for key, val in est.items():
                print(
                    f"  {key}: ≤ {val.get('total_upper_bound')} "
                    f"(agent={val.get('agent')}, defense={val.get('defense')})"
                )
    return 0


def cmd_agentdojo(args: argparse.Namespace) -> int:
    from llm_tester.agent.agentdojo_adapter import (
        format_agentdojo_summary,
        mock_agentdojo_suite,
        run_agentdojo_suite,
        run_real_agentdojo_suite,
        try_import_agentdojo,
    )
    from llm_tester.agent.provider_adapter import build_llm_chat

    provider_name = args.provider
    llm = None
    if provider_name and provider_name.lower() not in ("mock", "none", "scripted", "ci"):
        llm = build_llm_chat(
            provider_name,
            args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=None if args.no_seed else args.seed,
            max_retries=args.max_retries,
            num_thread=getattr(args, "num_thread", None),
        )

    if args.real:
        if try_import_agentdojo() is None:
            print(
                "AgentDojo package not installed. Install with: "
                "pip install 'llm-tester[agentdojo]'  "
                "(or use --suite mock without --real).",
                file=sys.stderr,
            )
            return 1
        # Default --real LLM is the offline scripted mock (no live API).
        effective_provider = provider_name or "mock"
        sample_seed = None if args.no_seed else args.seed
        try:
            summary, results = run_real_agentdojo_suite(
                args.suite,
                defense_name=args.defense,
                llm_provider=llm,
                provider_name=effective_provider,
                model=args.model,
                temperature=args.temperature,
                policy_path=args.policy,
                limit=args.limit,
                seed=sample_seed,
                benchmark_version=args.benchmark_version,
                attack_name=args.attack,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        if args.suite not in ("mock", "default", "ci"):
            print(
                f"Unknown offline suite {args.suite!r}. Use --suite mock, or "
                f"--real with an upstream suite name.",
                file=sys.stderr,
            )
            return 2
        tasks = mock_agentdojo_suite(limit=args.limit)
        summary, results = run_agentdojo_suite(
            tasks,
            defense_name=args.defense,
            llm_provider=llm,
            policy_path=args.policy,
        )

    run_dir = prepare_run_dir(
        args,
        "agentdojo",
        name_parts=["agentdojo", args.suite, args.defense],
    )
    with open_results_writer(run_dir, append=False) as writer:
        for rec in results:
            writer.write(rec.to_dict())

    metrics_dict = summary.to_metrics()
    metrics_dict["command"] = "agentdojo"
    metrics_dict["suite"] = args.suite
    metrics_dict["defense"] = args.defense
    metrics_dict["real"] = bool(args.real)
    metrics_dict["attack"] = getattr(args, "attack", None) if args.real else None
    config = {
        "command": "agentdojo",
        "suite": args.suite,
        "defense": args.defense,
        "real": bool(args.real),
        "limit": args.limit,
        "sample_seed": None if args.no_seed else args.seed,
        "provider": provider_name or ("mock" if args.real else None),
        "model": args.model,
        "num_thread": getattr(args, "num_thread", None),
        "attack": getattr(args, "attack", None) if args.real else None,
        "benchmark_version": args.benchmark_version if args.real else None,
    }
    finalize_run_dir(
        run_dir,
        command="agentdojo",
        config=config,
        metrics=metrics_dict,
        argv=sys.argv[1:],
        provider=provider_name,
        model=args.model,
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    print(f"Run artefacts written to {run_dir}")
    if args.json:
        print(json.dumps(metrics_dict, indent=2))
    else:
        print(
            format_agentdojo_summary(
                summary,
                title=f"agentdojo suite={args.suite} defense={args.defense}",
            )
        )
    return 0


def cmd_baselines_eval(args: argparse.Namespace) -> int:
    from llm_tester.agent.eval import evaluate_traces, format_metrics
    from llm_tester.agent.provider_adapter import build_llm_chat
    from llm_tester.agent.traces import TraceSchemaError, resolve_traces
    from llm_tester.baselines.registry import build_baseline

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

    llm = None
    if args.provider:
        llm = build_llm_chat(
            args.provider,
            args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=None if args.no_seed else args.seed,
            max_retries=args.max_retries,
            num_thread=getattr(args, "num_thread", None),
        )
    elif args.baseline == "task_shield":
        print(
            "Warning: task_shield without --provider will skip (ALLOW all). "
            "Use --provider ollama --model qwen2.5:3b for real runs.",
            file=sys.stderr,
        )

    sentinel_kwargs: Dict[str, Any] = {}
    if args.baseline == "sentinel":
        from llm_tester.sentinel.policy import load_policy

        pol = load_policy(args.policy)
        pol.enable_s4 = False  # offline default; enable via provider + future flag
        sentinel_kwargs["policy"] = pol

    guard = build_baseline(
        args.baseline,
        llm_provider=llm,
        spotlight_mode=args.spotlight_mode,
        prompt_guard_model=args.prompt_guard_model,
        prompt_guard_threshold=args.prompt_guard_threshold,
        sentinel_kwargs=sentinel_kwargs or None,
    )
    if getattr(guard, "skip_reason", None) and not getattr(guard, "available", True):
        print(f"Note: {guard.skip_reason}", file=sys.stderr)

    run_dir = prepare_run_dir(
        args,
        "baselines-eval",
        name_parts=["baselines-eval", args.baseline, args.dataset],
    )
    metrics, records = evaluate_traces(traces, guard, collect_records=True)
    with open_results_writer(run_dir, append=False) as writer:
        for rec in records:
            writer.write(rec)

    metrics_dict = metrics.to_dict()
    metrics_dict["dataset"] = args.dataset
    metrics_dict["command"] = "baselines-eval"
    metrics_dict["baseline"] = args.baseline
    metrics_dict["baseline_available"] = bool(getattr(guard, "available", True))
    last_err = getattr(guard, "last_error", None)
    if last_err:
        metrics_dict["last_error"] = last_err
    config: Dict[str, Any] = {
        "command": "baselines-eval",
        "baseline": args.baseline,
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "limit": args.limit,
        "spotlight_mode": args.spotlight_mode,
        "prompt_guard_model": args.prompt_guard_model,
        "prompt_guard_threshold": args.prompt_guard_threshold,
        "provider": args.provider,
        "model": args.model,
        "num_thread": getattr(args, "num_thread", None),
        "baseline_available": bool(getattr(guard, "available", True)),
        "skip_reason": getattr(guard, "skip_reason", None),
        "errors": metrics.n_errors,
        "n_errors": metrics.n_errors,
    }
    finalize_run_dir(
        run_dir,
        command="baselines-eval",
        config=config,
        metrics=metrics_dict,
        argv=sys.argv[1:],
        provider=args.provider,
        model=args.model,
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    print(f"Run artefacts written to {run_dir}")
    if metrics.n_errors:
        print(
            f"ERROR: baseline reported {metrics.n_errors} decision_error case(s). "
            f"Last: {last_err or '(see metrics.n_errors)'}",
            file=sys.stderr,
        )
    if args.json:
        print(json.dumps(metrics_dict, indent=2))
    else:
        print(
            format_metrics(
                metrics, title=f"baselines-eval baseline={args.baseline} dataset={args.dataset}"
            )
        )
    # Non-zero exit when every case errored (silent-ALLOW regression).
    if metrics.n and metrics.n_errors >= metrics.n:
        return 3
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    from llm_tester.agent.ablation import (
        format_ablation_table,
        pick_primary_cell,
        run_ablation_matrix,
    )
    from llm_tester.agent.traces import TraceSchemaError, resolve_traces
    from llm_tester.sentinel.policy import load_policy

    if args.write_adaptive:
        from llm_tester.agent.adaptive.generate import materialize_default_dataset

        path = materialize_default_dataset(
            Path(args.dataset_path) if args.dataset_path else None
        )
        print(f"Wrote adaptive dataset: {path}")
        if args.dataset != "adaptive" and args.dataset_path is None:
            # Still continue if user only wanted materialization? Prefer continue.
            pass

    try:
        if args.dataset == "adaptive":
            from llm_tester.agent.adaptive.generate import (
                DEFAULT_ADAPTIVE_PATH,
                load_adaptive_traces,
            )

            path = args.dataset_path or DEFAULT_ADAPTIVE_PATH
            traces = load_adaptive_traces(path)
            if args.limit is not None:
                traces = traces[: args.limit]
        else:
            traces = resolve_traces(
                args.dataset,
                path=args.dataset_path,
                limit=args.limit,
                download=False,
            )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except TraceSchemaError as exc:
        print(f"Trace schema error: {exc}", file=sys.stderr)
        return 2

    if not traces:
        print("No traces loaded.", file=sys.stderr)
        return 1

    aggs = [a.strip() for a in str(args.aggregations).split(",") if a.strip()]
    policy = load_policy(args.policy)
    result = run_ablation_matrix(
        traces,
        policy=policy,
        aggregations=aggs,
        include_checkpoint_ablation=not args.no_checkpoint_ablation,
    )

    run_dir = prepare_run_dir(
        args,
        "ablation",
        name_parts=["ablation", args.dataset],
    )
    # metrics.json = full-system default cell (full|noisy_or|T0+T1), NOT S1-only.
    # Complete matrix → ablation_matrix.json / ablation_summary.csv.
    primary_cell = pick_primary_cell(result)
    primary = dict(primary_cell["metrics"]) if primary_cell else {}
    primary["command"] = "ablation"
    primary["dataset"] = args.dataset
    primary["n_cells"] = result["n_cells"]
    primary["ablation_cell"] = primary_cell["name"] if primary_cell else None
    primary["metrics_scope"] = "primary_full_system"
    primary["matrix_path"] = "ablation_matrix.json"
    primary["note"] = (
        "metrics.json is the full|noisy_or|T0+T1 cell for index/compare; "
        "see ablation_matrix.json and ablation_summary.csv for every cell."
    )
    matrix_path = run_dir / "ablation_matrix.json"
    matrix_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    config = {
        "command": "ablation",
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "limit": args.limit,
        "aggregations": aggs,
        "checkpoint_ablation": not args.no_checkpoint_ablation,
        "n_cells": result["n_cells"],
        "cell_names": [c["name"] for c in result["cells"]],
        "primary_cell": primary_cell["name"] if primary_cell else None,
        "metrics_scope": "primary_full_system",
    }
    finalize_run_dir(
        run_dir,
        command="ablation",
        config=config,
        metrics=primary,
        argv=sys.argv[1:],
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    # Also drop a compact CSV for thesis tables.
    csv_path = run_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("name,stages,aggregation,checkpoints,tpr,fpr,f1,asr_with_guard\n")
        for c in result["cells"]:
            m = c["metrics"]
            fh.write(
                f"{c['name']},{c['stages']},{c['aggregation']},{c['checkpoints']},"
                f"{m.get('tpr')},{m.get('fpr')},{m.get('f1')},{m.get('asr_with_guard')}\n"
            )

    print(f"Run artefacts written to {run_dir}")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_ablation_table(result))
    return 0

def cmd_guard_calibrate(args: argparse.Namespace) -> int:
    from llm_tester.agent.calibrate import (
        parse_seeds_arg,
        run_calibration_multi_seed,
        write_calibration_report,
    )
    from llm_tester.agent.traces import TraceSchemaError, resolve_traces
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

    if len(traces) < 4:
        print("Need at least 4 traces to calibrate.", file=sys.stderr)
        return 1

    try:
        seeds = parse_seeds_arg(
            args.seeds, seed=args.seed, n_seeds=args.n_seeds
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_dir = prepare_run_dir(
        args,
        "guard-calibrate",
        name_parts=["guard-calibrate", args.dataset, args.split_by],
    )

    policy = load_policy(args.policy)
    policy.enable_s4 = False
    result = run_calibration_multi_seed(
        traces,
        seeds=seeds,
        base_policy=policy,
        strategy=args.split_by,
        train_frac=args.train_frac,
        objective=args.objective,
        max_fpr=args.max_fpr,
    )
    out_path = args.out or (run_dir / "calibration.json")
    write_calibration_report(result, out_path)

    held = result["heldout_metrics"]
    metrics_dict = dict(held)
    metrics_dict["command"] = "guard-calibrate"
    metrics_dict["dataset"] = args.dataset
    metrics_dict["train_metrics"] = result["train_metrics"]
    metrics_dict["calibrated_policy"] = result["calibrated_policy"]
    metrics_dict["seeds"] = result["seeds"]
    metrics_dict["heldout_aggregate"] = result.get("heldout_aggregate")

    config: Dict[str, Any] = {
        "command": "guard-calibrate",
        "dataset": args.dataset,
        "dataset_path": str(args.dataset_path) if args.dataset_path else None,
        "limit": args.limit,
        "policy": str(args.policy) if args.policy else None,
        "objective": args.objective,
        "max_fpr": args.max_fpr,
        "seeds": result["seeds"],
        "split": result["split"],
        "calibrated_policy": result["calibrated_policy"],
        "search": result.get("search"),
        "heldout_aggregate": result.get("heldout_aggregate"),
    }
    finalize_run_dir(
        run_dir,
        command="guard-calibrate",
        config=config,
        metrics=metrics_dict,
        argv=sys.argv[1:],
        extra_sha=parse_extra_sha(args.extra_sha),
    )
    # Status always on stderr so --json stdout is pure JSON.
    print(f"Run artefacts written to {run_dir}", file=sys.stderr)

    if args.json:
        printable = {
            k: v
            for k, v in result.items()
            if k not in ("train_report", "heldout_report", "aggregate_report")
        }
        print(json.dumps(printable, indent=2))
    else:
        print(
            f"=== guard-calibrate dataset={args.dataset} "
            f"split_by={args.split_by} seeds={result['seeds']} ==="
        )
        if result.get("aggregate_report"):
            print(result["aggregate_report"])
        print(result["train_report"])
        print(result["heldout_report"])
        print(f"\nWrote split + metrics to {out_path}", file=sys.stderr)
        print(
            "Note: thresholds tuned on train only; held-out numbers are the "
            "unbiased estimate. Preliminary — not published results."
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
    if getattr(args, "config", None):
        argv.extend(["--config", str(args.config)])
    if args.policy:
        argv.extend(["--policy", str(args.policy)])
    if args.log:
        argv.extend(["--log", str(args.log)])
    if getattr(args, "audit_log", None):
        argv.extend(["--audit-log", str(args.audit_log)])
    if args.user_task:
        argv.extend(["--user-task", args.user_task])
    if getattr(args, "threshold", None) is not None:
        argv.extend(["--threshold", str(args.threshold)])
    if getattr(args, "enable_s4", False):
        argv.append("--enable-s4")
    if getattr(args, "provider", None):
        argv.extend(["--provider", args.provider])
    if getattr(args, "model", None):
        argv.extend(["--model", args.model])
    if getattr(args, "transport", None):
        argv.extend(["--transport", args.transport])
    if getattr(args, "listen", None):
        argv.extend(["--listen", args.listen])
    argv.extend(args.command)

    llm = None
    if getattr(args, "enable_s4", False) and getattr(args, "provider", None):
        from llm_tester.agent.provider_adapter import build_llm_chat

        llm = build_llm_chat(args.provider, args.model)
    return proxy_main(argv, llm_provider=llm)


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
    if args.command == "baselines-eval":
        return cmd_baselines_eval(args)
    if args.command == "agentdojo":
        return cmd_agentdojo(args)
    if args.command == "ablation":
        return cmd_ablation(args)
    if args.command == "guard-calibrate":
        return cmd_guard_calibrate(args)
    if args.command == "grey-zone-rebuild":
        return cmd_grey_zone_rebuild(args)
    if args.command == "import-mcp-sim":
        return cmd_import_mcp_sim(args)
    if args.command == "adaptive-attack":
        return cmd_adaptive_attack(args)
    if args.command == "mcp-scenario-eval":
        return cmd_mcp_scenario_eval(args)
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
