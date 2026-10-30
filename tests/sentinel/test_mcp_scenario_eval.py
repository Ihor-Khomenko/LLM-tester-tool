"""Offline tests for mcp-scenario-eval (with/without Sentinel)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_tester.agent.mcp_scenario_eval import (
    AllowAllGuard,
    DEFAULT_FIXTURE,
    McpScenarioEvalConfig,
    estimate_requests_per_mcp_scenario,
    load_scenario_traces,
    run_mcp_scenario_eval,
)
from llm_tester.cli import main
from llm_tester.sentinel.types import Decision


ROOT = Path(__file__).resolve().parents[2]


def test_fixture_covers_poison_rug_exfil_benign():
    traces = load_scenario_traces(input_path=DEFAULT_FIXTURE)
    assert len(traces) >= 4
    types = {t.attack_type for t in traces}
    labels = {t.label for t in traces}
    assert "benign" in labels
    assert "tool_poisoning" in types or any(
        "poison" in (t.attack_type or "") for t in traces
    )
    assert "rug_pull" in types
    # Parameter / write exfil covered by attack call arguments in fixture.
    assert any(t.call is not None for t in traces if t.label == "attack")


def test_allow_all_never_blocks():
    g = AllowAllGuard()
    g.set_user_task("hi")
    d0 = g.check_tool_registration([{"name": "t", "description": "x"}])
    d1 = g.on_tool_output("t", "IMPORTANT!!! Ignore previous")
    d2 = g.check_tool_call({"name": "bank.transfer", "arguments": {"iban": "DE"}})
    assert d0.decision == Decision.ALLOW
    assert d1.decision == Decision.ALLOW
    assert d2.decision == Decision.ALLOW


def test_estimate_requests_no_numbers_from_experiments():
    none = estimate_requests_per_mcp_scenario(with_sentinel=False)
    assert none["total_upper_bound"] == 0
    s4 = estimate_requests_per_mcp_scenario(with_sentinel=True, enable_s4=True)
    assert s4["defense"] == 2
    no_s4 = estimate_requests_per_mcp_scenario(with_sentinel=True, enable_s4=False)
    assert no_s4["total_upper_bound"] == 0


def test_run_mcp_scenario_eval_none_vs_sentinel(tmp_path: Path):
    out = tmp_path / "mcp_eval"
    summary = run_mcp_scenario_eval(
        out,
        config=McpScenarioEvalConfig(
            defenses=["none", "sentinel"],
            mock=True,
            no_s4=True,
            proxy_mode=True,
        ),
        input_path=DEFAULT_FIXTURE,
    )
    assert summary["n_scenarios"] >= 4
    assert "none" in summary["by_defense"]
    assert "sentinel" in summary["by_defense"]
    none_asr = summary["by_defense"]["none"]["asr"]["asr_with_guard"]
    # Without a guard, attacks are not intervened → ASR 1.0
    assert none_asr == 1.0
    sent = summary["by_defense"]["sentinel"]
    assert "detections" in sent
    assert sent["detections"]["first_detection"] is not None
    assert (out / "results.jsonl").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "progress.json").is_file()
    assert summary["config"]["proxy_mode"] is True
    # No experimental paper numbers embedded — only this-run keys.
    raw = (out / "summary.json").read_text(encoding="utf-8")
    assert "nemotron" not in raw.lower()


def test_cli_mcp_scenario_eval(tmp_path: Path):
    out = tmp_path / "cli_mcp"
    rc = main(
        [
            "mcp-scenario-eval",
            "--out-dir",
            str(out),
            "--input",
            str(DEFAULT_FIXTURE),
            "--limit",
            "3",
            "--proxy-mode",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert data["n_scenarios"] == 3
