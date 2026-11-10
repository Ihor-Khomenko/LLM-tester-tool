"""Analyzer over a tiny real_mcp fixture (no corpora / Ollama)."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "analyzer_run"


def test_analyzer_real_mcp_fixture(research_runner):
    runs = research_runner["runs"]
    # Copy fixture day/job tree into OBERIH_RUNS_AUTO
    for day in FIXTURE.iterdir():
        if day.is_dir():
            shutil.copytree(day, runs / day.name)

    if "analyzer" in sys.modules:
        del sys.modules["analyzer"]
    if "real_mcp_report" in sys.modules:
        del sys.modules["real_mcp_report"]
    analyzer = importlib.import_module("analyzer")

    out = analyzer.main()
    assert "real_mcp" in out
    assert out["n_jobs_scored"] >= 1

    # Attack ASR on none arm should be 1/1; oberih should show 0 successes.
    rm = out["real_mcp"]
    none_key = next(k for k in rm if k.endswith("|none"))
    oberih_key = next(k for k in rm if k.endswith("|oberih"))
    assert rm[none_key]["succ"] == 1
    assert rm[none_key]["n_att"] == 1
    assert rm[oberih_key]["succ"] == 0
    assert rm[oberih_key]["n_ben"] == 1
    assert rm[oberih_key]["fp"] == 0

    runner = research_runner["runner"]
    summary = json.loads((runner / "summary.json").read_text())
    assert summary["real_mcp"]
    assert (runner / "summary.md").is_file()
    assert (runner / "findings.md").is_file()
    assert (runner / "proposals.md").is_file()
