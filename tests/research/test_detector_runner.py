"""Supervisor / planner wiring for external detectors + detector-only jobs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load(name: str, research_runner):
    if name in sys.modules:
        del sys.modules[name]
    return importlib.import_module(name)


def test_steps_for_external_detectors(research_runner):
    supervisor = _load("supervisor", research_runner)
    pol = research_runner["policy"]
    job = {
        "id": "p-att-det",
        "kind": "live",
        "model": "granite3-moe:3b",
        "corpus": "v6-hard",
        "arms": [
            "none",
            "oberih",
            "prompt_guard2",
            "protectai_deberta",
            "llama_guard3",
            "qwen3_guard",
        ],
        "langs": ["uk", "en"],
        "labels": ["attack"],
        "temperature": 0.7,
        "seed": 3101,
        "n_variants": 2,
        "scenarios": [],
    }
    steps = supervisor.steps_for(job, pol, Path("/tmp/out-det"))
    names = [s[0] for s in steps]
    assert "base" in names
    for arm in ("prompt_guard2", "protectai_deberta", "llama_guard3", "qwen3_guard"):
        assert arm in names


def test_steps_for_detector_kind(research_runner):
    supervisor = _load("supervisor", research_runner)
    pol = research_runner["policy"]
    job = {
        "id": "i-det-synthetic-s42",
        "kind": "detector",
        "model": "detector",
        "corpus": "synthetic",
        "arms": ["oberih_s1", "prompt_guard2"],
        "langs": ["en"],
        "labels": ["attack", "benign"],
        "temperature": 0.0,
        "seed": 42,
        "limit": 4,
    }
    steps = supervisor.steps_for(job, pol, Path("/tmp/out-idet"))
    assert len(steps) == 1
    assert steps[0][0] == "detector"
    assert steps[0][4] == "detector"
    cmd = steps[0][2]
    assert "run_detector_eval.py" in " ".join(str(x) for x in cmd)
    assert "--corpus" in cmd and "synthetic" in cmd


def test_planner_gen_i_detector(research_runner):
    planner = research_runner["planner"]
    pol = research_runner["policy"]
    pol.setdefault("planner", {})
    pol["planner"]["i_detector"] = {
        "corpora": ["synthetic", "bipia"],
        "arms": ["oberih_s1", "prompt_guard2"],
        "langs": ["en"],
        "seeds": [42],
        "threshold": 0.5,
        "split": "test",
    }
    # Write updated policy so planner.policy() sees it if it re-reads; gen_i takes pol arg.
    j = planner.gen_i({}, pol, {})
    assert j is not None
    assert j["kind"] == "detector"
    assert j["id"].startswith("i-det-")
    assert j["corpus"] in ("synthetic", "bipia")
    assert "prompt_guard2" in j["arms"]


def test_detector_gates_canonical():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "runner" / "lib"))
    if "detector_gates" in sys.modules:
        del sys.modules["detector_gates"]
    DG = importlib.import_module("detector_gates")
    assert DG.canonical_detector_arm("pg2") == "prompt_guard2"
    assert DG.canonical_detector_arm("protectai") == "protectai_deberta"
    assert DG.canonical_detector_arm("llama_guard") == "llama_guard3"
    assert DG.canonical_detector_arm("qwen3guard") == "qwen3_guard"
    assert DG.is_external_detector_arm("qwen3_guard")
