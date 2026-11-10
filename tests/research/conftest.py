"""Fixtures for research/runner offline tests (isolated OBERIH_* roots)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
RUNNER_SRC = REPO / "research" / "runner"
POLICY_SRC = RUNNER_SRC / "policy.yaml"


@pytest.fixture
def research_runner(tmp_path, monkeypatch):
    """Isolated runner workspace with a copy of policy.yaml (no legacy wait / ollama)."""
    ws = tmp_path / "ws"
    runner = ws / "runner"
    for sub in ("pending", "running", "done", "failed"):
        (runner / "queue" / sub).mkdir(parents=True)
    (runner / "lib").mkdir(parents=True)
    (runner / "logs").mkdir()
    (ws / "real_mcp").mkdir()
    runs = tmp_path / "runs_auto"
    runs.mkdir()

    pol = yaml.safe_load(POLICY_SRC.read_text()) or {}
    pol["legacy_wait"] = []
    pol.setdefault("guards", {})
    pol["guards"]["require_ollama"] = False
    pol["guards"]["battery_min_pct"] = 0
    pol["guards"]["min_free_gb"] = 0
    pol["guards"]["max_swap_mb"] = 10**9
    # Point llm_tester at this repo for any child imports.
    pol["llm_tester_dir"] = str(REPO)
    (runner / "policy.yaml").write_text(yaml.safe_dump(pol))

    monkeypatch.setenv("OBERIH_RUNNER_ROOT", str(runner))
    monkeypatch.setenv("OBERIH_TESTS_ROOT", str(ws))
    monkeypatch.setenv("RESEARCH_ROOT", str(ws))
    monkeypatch.setenv("OBERIH_RUNS_AUTO", str(runs))
    monkeypatch.setenv("LLM_TESTER_ROOT", str(REPO))
    monkeypatch.setenv("PYTHONPATH", str(REPO))

    lib = str(RUNNER_SRC / "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)

    # Fresh imports so module-level RUNNER/ROOT pick up env.
    for name in list(sys.modules):
        if name in ("common", "planner", "supervisor", "analyzer", "real_mcp_report") or name.startswith(
            ("common.", "planner.", "supervisor.", "analyzer.")
        ):
            del sys.modules[name]

    common = importlib.import_module("common")
    planner = importlib.import_module("planner")
    return {
        "ws": ws,
        "runner": runner,
        "runs": runs,
        "common": common,
        "planner": planner,
        "policy": pol,
        "lib": lib,
    }
