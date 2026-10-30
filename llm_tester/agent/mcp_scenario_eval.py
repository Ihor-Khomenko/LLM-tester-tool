"""
Evaluate mcp-security-research scenarios with and without MCP-Sentinel.

Offline / CI path: import schema-1.0 (or v1) JSONL → ``evaluate_traces`` under
``none`` (always-allow) and ``sentinel`` (S1–S3; optional S4 mock). Live boxes
can point ``--research-root`` at a checkout or feed proxy-mode audits; this
module never makes network calls unless the caller attaches a real LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from llm_tester.agent.eval import evaluate_traces
from llm_tester.agent.import_mcp_sim import convert_records, load_mcpsr_jsonl
from llm_tester.agent.traces import TraceStep, load_jsonl_traces, parse_trace_record
from llm_tester.baselines.protocol import BaseGuard, allow_decision
from llm_tester.sentinel.guard import build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy, load_policy
from llm_tester.sentinel.types import Checkpoint, ToolCall

# Bundled offline fixture (tool poisoning, rug-pull, IPI, write/exfil, benign).
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "agent"
    / "mcp_sim_import_fixture.jsonl"
)

# Documentation-only upper bounds (no measured experimental numbers).
EST_REQUESTS_PER_SCENARIO_NO_S4 = 0  # S1–S3 offline: 0 LLM calls
EST_REQUESTS_PER_SCENARIO_WITH_S4 = 2  # ~S4 judge calls when enabled
EST_REQUESTS_PROXY_OVERHEAD = 0  # proxy itself does not call LLMs without S4


class AllowAllGuard(BaseGuard):
    """Passthrough defense for the ``none`` condition (ASR without Sentinel)."""

    name = "none"

    def check_tool_registration(self, tools):  # type: ignore[override]
        return allow_decision(
            checkpoint=Checkpoint.T0.value,
            stage="none",
            reasons=["none:allow"],
        )

    def on_tool_output(self, source: str, content: str, *, sanitize: bool = True):
        return allow_decision(
            checkpoint=Checkpoint.T1.value,
            stage="none",
            reasons=["none:allow"],
            sanitized_content=content if sanitize else None,
        )

    def check_tool_call(
        self,
        call: Union[ToolCall, Dict[str, Any]],
        *,
        user_request: Optional[str] = None,
    ):
        return allow_decision(
            checkpoint=Checkpoint.T2.value,
            stage="none",
            reasons=["none:allow"],
        )


@dataclass
class McpScenarioEvalConfig:
    defenses: List[str] = field(default_factory=lambda: ["none", "sentinel"])
    mock: bool = True
    enable_s4: bool = False
    s4_mock: bool = False
    no_s4: bool = True  # default offline: S1–S3 only
    limit: Optional[int] = None
    seed: int = 42
    # When True, record that results are comparable to proxy-mode T0/T1/T2.
    proxy_mode: bool = False


def estimate_requests_per_mcp_scenario(
    *,
    with_sentinel: bool = True,
    enable_s4: bool = False,
) -> Dict[str, int]:
    """
    Upper-bound request estimate for one scenario.

    Trace replay without S4 uses zero LLM calls. With S4, plan ~2 judge calls
    per scenario that reaches the grey zone. Agent-side LLM is separate
    (agent-bench / live proxy clients) and not counted here.
    """
    if not with_sentinel:
        return {"defense": 0, "agent": 0, "total_upper_bound": 0}
    defense = EST_REQUESTS_PER_SCENARIO_WITH_S4 if enable_s4 else EST_REQUESTS_PER_SCENARIO_NO_S4
    return {
        "defense": defense,
        "agent": 0,
        "total_upper_bound": defense,
    }


def load_scenario_traces(
    *,
    input_path: Optional[Path] = None,
    research_root: Optional[Path] = None,
    limit: Optional[int] = None,
    lang: str = "en",
    scenario_ids: Optional[Sequence[str]] = None,
) -> List[TraceStep]:
    """
    Load TraceSteps from MSR schema 1.0, v1 JSONL, or a research checkout.

    Defaults to the bundled ``mcp_sim_import_fixture.jsonl``.
    """
    if research_root is not None:
        from llm_tester.agent.import_mcp_sim import export_from_research_root

        exported = export_from_research_root(
            Path(research_root),
            lang=lang,
            scenario_ids=scenario_ids,
        )
        records = load_mcpsr_jsonl(exported)
        cases, _ = convert_records(records)
        traces = [parse_trace_record(c) for c in cases]
    else:
        path = Path(input_path) if input_path else DEFAULT_FIXTURE
        if not path.is_file():
            raise FileNotFoundError(f"MCP scenario JSONL not found: {path}")
        # load_jsonl_traces auto-converts schema 1.0 via mcpsr.
        traces = load_jsonl_traces(path)

    if scenario_ids:
        want = {str(x) for x in scenario_ids}
        traces = [t for t in traces if str(t.id) in want]
    if limit is not None:
        traces = traces[: max(0, int(limit))]
    return traces


def _build_defense(
    name: str,
    *,
    config: McpScenarioEvalConfig,
    policy: Optional[SentinelPolicy] = None,
    llm: Any = None,
) -> Any:
    n = (name or "none").lower().strip()
    if n in ("none", "off", "allow"):
        return AllowAllGuard()
    if n in ("sentinel", "sentinel_no_s4", "s4_only"):
        pol = policy or load_policy(None)
        if n == "sentinel_no_s4" or (config.no_s4 and not config.enable_s4 and not config.s4_mock):
            pol = pol.with_ablation(no_s4=True)
        elif n == "s4_only":
            pol = pol.with_ablation(only_s4=True)
            pol.tau_low = -1.0
            pol.tau_high = 2.0
        elif config.enable_s4 or config.s4_mock:
            pol.enable_s4 = True
        else:
            pol.enable_s4 = False
        if config.s4_mock and pol.enable_s4 and llm is None:
            from llm_tester.agent.s4_mock import MockS4Chat

            llm = MockS4Chat(seed=config.seed)
        return build_sentinel(policy=pol, llm_provider=llm)
    raise ValueError(
        f"Unknown defense {name!r}; use none|sentinel|sentinel_no_s4|s4_only"
    )


def _utility_from_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Offline utility proxies from confusion matrices.

    * ``utility_benign`` — share of benign cases not intervened (≈ 1 − FPR).
    * ``utility_under_attack`` — share of attack cases not hard-blocked
      (task may still complete; offline traces have no agent success bit).
    """
    n_benign = int(m.get("n_benign") or 0)
    n_attack = int(m.get("n_attack") or 0)
    any_i = m.get("any_intervention") or {}
    block = m.get("block_only") or {}
    tn = int(any_i.get("tn") or 0)
    fp = int(any_i.get("fp") or 0)
    # Attacks not hard-blocked
    fn_block = int(block.get("fn") or 0)
    return {
        "utility_benign": (tn / n_benign) if n_benign else None,
        "utility_under_attack": (fn_block / n_attack) if n_attack else None,
        "fpr_any_intervention": any_i.get("fpr"),
        "n_benign": n_benign,
        "n_attack": n_attack,
    }


def run_mcp_scenario_eval(
    out_dir: Path,
    *,
    config: Optional[McpScenarioEvalConfig] = None,
    input_path: Optional[Path] = None,
    research_root: Optional[Path] = None,
    policy_path: Optional[Path] = None,
    scenario_ids: Optional[Sequence[str]] = None,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Run the with/without Sentinel matrix; write ``results.jsonl`` + ``summary.json``.

    Does not embed historical experimental numbers — only this-run counters.
    """
    cfg = config or McpScenarioEvalConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = load_scenario_traces(
        input_path=input_path,
        research_root=research_root,
        limit=cfg.limit,
        lang=lang,
        scenario_ids=scenario_ids,
    )
    if not traces:
        raise ValueError("No MCP scenarios loaded")

    policy = load_policy(policy_path) if policy_path else load_policy(None)
    results_path = out_dir / "results.jsonl"
    by_defense: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    if results_path.is_file():
        results_path.unlink()

    for defense in cfg.defenses:
        guard = _build_defense(defense, config=cfg, policy=policy)
        metrics, records = evaluate_traces(traces, guard, collect_records=True)
        mdict = metrics.to_dict()
        util = _utility_from_metrics(mdict)
        entry = {
            "defense": defense,
            "metrics": mdict,
            "utility": util,
            "asr": {
                "asr_without_guard": mdict.get("asr_without_guard"),
                "asr_with_guard": mdict.get("asr_with_guard"),
                "asr_block_only": mdict.get("asr_block_only"),
            },
            "detections": {
                "tpr": mdict.get("tpr"),
                "fpr": mdict.get("fpr"),
                "first_detection": mdict.get("first_detection"),
                "first_block": mdict.get("first_block"),
                "by_checkpoint": mdict.get("by_checkpoint"),
                "by_attack_type": mdict.get("by_attack_type"),
            },
            "llm_calls_per_case": mdict.get("llm_calls_per_case"),
            "proxy_mode": bool(cfg.proxy_mode),
        }
        by_defense[defense] = entry
        for rec in records:
            row = {
                "defense": defense,
                "id": rec.get("id"),
                "label": rec.get("label"),
                "attack_type": rec.get("attack_type"),
                "intervened": rec.get("intervened"),
                "blocked": rec.get("blocked"),
                "first_intervention": rec.get("first_intervention"),
                "first_block": rec.get("first_block"),
                "risk_score": rec.get("risk_score"),
            }
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary: Dict[str, Any] = {
        "n_scenarios": len(traces),
        "scenario_ids": [t.id for t in traces],
        "defenses": list(cfg.defenses),
        "by_defense": by_defense,
        "config": {
            "mock": cfg.mock,
            "enable_s4": cfg.enable_s4,
            "s4_mock": cfg.s4_mock,
            "no_s4": cfg.no_s4,
            "proxy_mode": cfg.proxy_mode,
            "seed": cfg.seed,
            "limit": cfg.limit,
        },
        "request_estimates": {
            "none": estimate_requests_per_mcp_scenario(with_sentinel=False),
            "sentinel_no_s4": estimate_requests_per_mcp_scenario(
                with_sentinel=True, enable_s4=False
            ),
            "sentinel_with_s4": estimate_requests_per_mcp_scenario(
                with_sentinel=True, enable_s4=True
            ),
        },
        "results_path": str(results_path.resolve()),
        "note": (
            "Offline trace replay. proxy_mode marks T0/T1/T2 parity with "
            "llmtest sentinel-proxy; attach a live MCP server separately."
        ),
    }

    # Paired none vs sentinel ASR when both present.
    if "none" in by_defense and "sentinel" in by_defense:
        none_asr = by_defense["none"]["asr"].get("asr_with_guard")
        sent_asr = by_defense["sentinel"]["asr"].get("asr_with_guard")
        summary["comparison"] = {
            "asr_none": none_asr,
            "asr_sentinel": sent_asr,
            "utility_benign_none": by_defense["none"]["utility"].get("utility_benign"),
            "utility_benign_sentinel": by_defense["sentinel"]["utility"].get(
                "utility_benign"
            ),
        }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "progress.json").write_text(
        json.dumps(
            {
                "n_scenarios": len(traces),
                "defenses": list(cfg.defenses),
                "done": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
