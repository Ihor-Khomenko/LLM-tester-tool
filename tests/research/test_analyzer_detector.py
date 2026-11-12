"""Analyzer must summarise kind=detector jobs in a detector-only table (offline)."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "analyzer_detector"


def test_analyzer_detector_jobs_not_in_live_cells(research_runner, monkeypatch):
    runs = research_runner["runs"]
    for day in FIXTURE.iterdir():
        if day.is_dir():
            shutil.copytree(day, runs / day.name)

    for name in ("analyzer", "real_mcp_report"):
        if name in sys.modules:
            del sys.modules[name]
    analyzer = importlib.import_module("analyzer")
    # Detector corpora must never call live scenario targets().
    monkeypatch.setattr(
        analyzer,
        "targets",
        lambda corpus: (_ for _ in ()).throw(
            AssertionError(f"targets() should not run for detector corpus {corpus!r}")
        ),
    )

    out = analyzer.main()
    runner = research_runner["runner"]
    summary = json.loads((runner / "summary.json").read_text())
    md = (runner / "summary.md").read_text()

    # No phantom live cells (previously: n=0, errors=6 per arm).
    assert summary.get("cells") == {}
    assert out.get("cells") == {}

    det = summary.get("detector") or []
    assert len(det) == 1
    assert det[0]["job"] == "i-det-synthetic-s42"
    assert det[0]["corpus"] == "synthetic"
    assert len(det[0]["arms"]) == 6

    assert "## Detector-only (TPR / FPR, Wilson 95% CI)" in md
    assert "i-det-synthetic-s42" in md
    assert "oberih_s1" in md
    assert "50.0%" in md or "0.5" in md
    # Must not treat detector JSONL rows as live case_status errors.
    findings = (runner / "findings.md").read_text()
    assert "case_status=" not in findings
    assert "targets(synthetic) failed" not in findings
