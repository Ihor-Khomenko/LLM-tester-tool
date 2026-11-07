"""Compatibility: Оберіг / Oberih naming with legacy Sentinel aliases."""

from __future__ import annotations

import json
from pathlib import Path

from llm_tester.agent.mcp_scenario_eval import (
    DEFAULT_FIXTURE,
    McpScenarioEvalConfig,
    _build_defense,
    run_mcp_scenario_eval,
)
from llm_tester.agent.metrics_stats import summarize_adaptive_results
from llm_tester.baselines.registry import BASELINE_NAMES, build_baseline
from llm_tester.cli import main
from llm_tester.sentinel.branding import (
    DISPLAY_NAME_EN,
    DISPLAY_NAME_UK,
    MACHINE_ID,
    canonical_defense_id,
    find_oberih_defense_key,
    is_oberih_defense,
    product_blurb,
)
from llm_tester.sentinel.guard import OberihGuard, SentinelGuard, build_oberih, build_sentinel
from llm_tester.sentinel.policy import OberihPolicy, SentinelPolicy


def test_canonical_branding_constants():
    assert DISPLAY_NAME_UK == "Оберіг"
    assert DISPLAY_NAME_EN == "Oberih"
    assert MACHINE_ID == "oberih"
    assert "Оберіг" in product_blurb(uk=True)
    assert "Oberih" in product_blurb(uk=False)


def test_legacy_imports_and_aliases_identical():
    assert OberihGuard is SentinelGuard
    assert OberihPolicy is SentinelPolicy
    assert build_oberih is build_sentinel
    g1 = build_sentinel(enable_s4=False)
    g2 = build_oberih(enable_s4=False)
    assert type(g1) is type(g2) is SentinelGuard


def test_oberih_package_reexports():
    import llm_tester.oberih as oberih

    assert oberih.MACHINE_ID == "oberih"
    assert oberih.build_oberih is build_oberih
    assert oberih.SentinelGuard is SentinelGuard


def test_defense_alias_normalization():
    assert canonical_defense_id("sentinel") == "oberih"
    assert canonical_defense_id("oberih") == "oberih"
    assert canonical_defense_id("mcp-sentinel") == "oberih"
    assert canonical_defense_id("sentinel_no_s4") == "oberih_no_s4"
    assert canonical_defense_id("oberih_no_s4") == "oberih_no_s4"
    assert is_oberih_defense("sentinel")
    assert is_oberih_defense("oberih")
    assert not is_oberih_defense("spotlighting")


def test_build_defense_accepts_sentinel_and_oberih():
    cfg = McpScenarioEvalConfig(no_s4=True, enable_s4=False, s4_mock=False)
    g_legacy = _build_defense("sentinel", config=cfg)
    g_new = _build_defense("oberih", config=cfg)
    assert isinstance(g_legacy, SentinelGuard)
    assert isinstance(g_new, OberihGuard)
    g_no = _build_defense("sentinel_no_s4", config=cfg)
    assert isinstance(g_no, SentinelGuard)


def test_baseline_registry_accepts_both():
    assert "oberih" in BASELINE_NAMES
    assert "sentinel" in BASELINE_NAMES
    a = build_baseline("oberih")
    b = build_baseline("sentinel")
    assert type(a) is type(b)


def test_mcp_eval_default_defense_is_oberih_and_sentinel_alias_works(tmp_path: Path):
    out_new = tmp_path / "oberih_def"
    summary = run_mcp_scenario_eval(
        out_new,
        config=McpScenarioEvalConfig(
            defenses=["none", "oberih"],
            mode="replay",
            n_variants=1,
            limit=2,
            no_s4=True,
        ),
        input_path=DEFAULT_FIXTURE,
    )
    assert "oberih" in summary["by_defense"]
    assert summary["comparison"]["tpr_oberih"] == summary["comparison"]["tpr_sentinel"]
    assert summary["comparison"]["oberih_defense_key"] == "oberih"

    out_legacy = tmp_path / "sentinel_def"
    summary2 = run_mcp_scenario_eval(
        out_legacy,
        config=McpScenarioEvalConfig(
            defenses=["none", "sentinel"],
            mode="replay",
            n_variants=1,
            limit=2,
            no_s4=True,
        ),
        input_path=DEFAULT_FIXTURE,
    )
    assert "sentinel" in summary2["by_defense"]
    assert find_oberih_defense_key(summary2["by_defense"]) == "sentinel"
    assert "tpr_sentinel" in summary2["comparison"]
    assert "tpr_oberih" in summary2["comparison"]


def test_adaptive_summary_pairs_legacy_sentinel_rows():
    jobs = [
        {"job_id": "j1", "defense": "none", "attack_success": True, "try_successes": [True]},
        {"job_id": "j1", "defense": "sentinel", "attack_success": False, "try_successes": [False]},
        {"job_id": "j2", "defense": "none", "attack_success": True, "try_successes": [True]},
        {"job_id": "j2", "defense": "sentinel", "attack_success": False, "try_successes": [False]},
    ]
    # Default defense_b=oberih must still pair with legacy sentinel labels.
    out = summarize_adaptive_results(jobs)
    assert "mcnemar" in out
    assert out["mcnemar"]["n"] == 2
    assert "sentinel" in out["by_defense"]


def test_cli_sentinel_defense_and_oberih_proxy_alias(tmp_path: Path):
    import pytest

    out = tmp_path / "cli_legacy"
    rc = main(
        [
            "mcp-scenario-eval",
            "--out-dir",
            str(out),
            "--input",
            str(DEFAULT_FIXTURE),
            "--defense",
            "none",
            "--defense",
            "sentinel",
            "--limit",
            "1",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "sentinel" in data["by_defense"]

    # Subparser alias oberih-proxy → same parser as sentinel-proxy.
    with pytest.raises(SystemExit) as exc:
        main(["oberih-proxy", "--help"])
    assert exc.value.code == 0


def test_cli_default_defenses_use_oberih(tmp_path: Path):
    out = tmp_path / "cli_default"
    rc = main(
        [
            "mcp-scenario-eval",
            "--out-dir",
            str(out),
            "--input",
            str(DEFAULT_FIXTURE),
            "--limit",
            "1",
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert data["defenses"] == ["none", "oberih"]
