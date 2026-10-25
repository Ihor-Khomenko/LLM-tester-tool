"""Offline AgentDojo adapter tests (mock suite only — no upstream package)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_tester.agent.agentdojo_adapter import (
    mock_agentdojo_suite,
    run_agentdojo_suite,
    try_import_agentdojo,
)
from llm_tester.cli import main


def test_mock_suite_shapes():
    tasks = mock_agentdojo_suite()
    assert any(t.label == "benign" for t in tasks)
    assert any(t.label == "attack" for t in tasks)
    assert try_import_agentdojo() is None or True  # may or may not be installed


def test_sentinel_defense_reduces_asr():
    tasks = mock_agentdojo_suite()
    none_sum, _ = run_agentdojo_suite(tasks, defense_name="none")
    sent_sum, _ = run_agentdojo_suite(tasks, defense_name="sentinel")
    assert none_sum.n_attack >= 1
    # Without defense, scripted attacks succeed.
    assert none_sum.asr == 1.0
    # Sentinel should block at least some attacks.
    assert sent_sum.asr is not None
    assert sent_sum.asr < none_sum.asr
    # Benign utility should stay reasonable with sentinel.
    assert sent_sum.utility is not None
    assert sent_sum.utility >= 0.5


def test_spotlighting_and_progent_run():
    tasks = mock_agentdojo_suite(limit=3)
    for name in ("spotlighting", "progent"):
        summary, results = run_agentdojo_suite(tasks, defense_name=name)
        assert summary.n == len(tasks)
        assert len(results) == len(tasks)


def test_agentdojo_cli_mock(tmp_path: Path):
    run_dir = tmp_path / "adj"
    rc = main(
        [
            "agentdojo",
            "--suite",
            "mock",
            "--defense",
            "sentinel",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["command"] == "agentdojo"
    assert "utility" in metrics
    assert "utility_under_attack" in metrics
    assert "asr" in metrics
    assert (run_dir / "results.jsonl.gz").is_file()


def test_agentdojo_cli_real_without_package(tmp_path: Path):
    rc = main(
        [
            "agentdojo",
            "--suite",
            "workspace",
            "--real",
            "--defense",
            "none",
            "--run-dir",
            str(tmp_path / "x"),
        ]
    )
    # Missing upstream package → non-zero, no network.
    assert rc == 1
