"""Supervisor: fresh timeout per retry attempt; graded / prompt_guard2 arm routing."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_supervisor(research_runner):
    if "supervisor" in sys.modules:
        del sys.modules["supervisor"]
    return importlib.import_module("supervisor")


def test_steps_for_graded_and_prompt_guard2(research_runner):
    supervisor = _load_supervisor(research_runner)
    pol = research_runner["policy"]
    job = {
        "id": "p-att-x",
        "kind": "live",
        "model": "granite3-moe:3b",
        "corpus": "v6-hard",
        "arms": ["none", "oberih", "oberih_graded", "prompt_guard2"],
        "langs": ["uk", "en"],
        "labels": ["attack"],
        "temperature": 0.7,
        "seed": 3101,
        "n_variants": 2,
        "scenarios": [],
    }
    steps = supervisor.steps_for(job, pol, Path("/tmp/out"))
    names = [s[0] for s in steps]
    assert "base" in names
    assert "oberih_graded" in names
    assert "prompt_guard2" in names
    # defense strings passed to driver
    by_name = {s[0]: s for s in steps}
    assert "oberih_graded" in by_name["oberih_graded"][2]  # cmd list contains defense
    assert "prompt_guard2" in by_name["prompt_guard2"][2]


def test_deadline_ignores_prior_elapsed(research_runner, monkeypatch):
    """Regression: retry used to subtract elapsed_s and time out immediately."""
    supervisor = _load_supervisor(research_runner)
    # Inspect the formula used at the start of run_job by reading source contract:
    # deadline = now + timeout_s  (NOT timeout_s - elapsed_s)
    src = Path(supervisor.__file__).read_text()
    assert "timeout_s" in src
    assert "fresh" in src.lower() or "Each retry attempt gets a full fresh timeout" in src
    # Ensure the old subtract pattern is gone.
    assert "timeout_s\", 10800)) - int(job.get(\"elapsed_s\"" not in src
    assert '- int(job.get("elapsed_s"' not in src
