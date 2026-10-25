"""Offline AgentDojo adapter tests (mock suite + mocked real pipeline)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_tester.agent.agentdojo_adapter import (
    mock_agentdojo_suite,
    run_agentdojo_suite,
    sample_balanced_tasks,
    sample_real_suite_jobs,
    try_import_agentdojo,
)
from llm_tester.cli import main


def test_mock_suite_shapes():
    tasks = mock_agentdojo_suite()
    assert any(t.label == "benign" for t in tasks)
    assert any(t.label == "attack" for t in tasks)


def test_limit_samples_both_labels():
    tasks = mock_agentdojo_suite(limit=3)
    assert len(tasks) == 3
    assert any(t.label == "benign" for t in tasks)
    assert any(t.label == "attack" for t in tasks)
    # Balanced helper itself.
    all_tasks = mock_agentdojo_suite()
    sampled = sample_balanced_tasks(all_tasks, 2)
    assert len(sampled) == 2
    labels = {t.label for t in sampled}
    assert "benign" in labels and "attack" in labels


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
    with patch(
        "llm_tester.agent.agentdojo_adapter.try_import_agentdojo", return_value=None
    ):
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


@pytest.mark.skipif(try_import_agentdojo() is None, reason="agentdojo not installed")
def test_real_pipeline_mock_llm_limit_balanced(tmp_path: Path):
    """Real AgentDojo loop with scripted mock LLM (no live API)."""
    from llm_tester.agent.agentdojo_adapter import (
        load_agentdojo_suite,
        run_real_agentdojo_suite,
        sample_real_suite_jobs,
    )

    suite = load_agentdojo_suite("workspace", benchmark_version="v1.2.2")
    jobs = sample_real_suite_jobs(suite, limit=4)
    assert len(jobs) == 4
    assert any(j["label"] == "benign" for j in jobs)
    assert any(j["label"] == "attack" for j in jobs)

    summary, results = run_real_agentdojo_suite(
        "workspace",
        defense_name="none",
        provider_name="mock",
        limit=4,
        benchmark_version="v1.2.2",
        attack_name="important_instructions",
    )
    assert summary.n == 4
    assert summary.n_benign >= 1
    assert summary.n_attack >= 1
    assert len(results) == 4
    assert all(r.meta.get("llm_mode") == "mock" for r in results)


@pytest.mark.skipif(try_import_agentdojo() is None, reason="agentdojo not installed")
def test_real_pipeline_cli_mock_provider(tmp_path: Path):
    run_dir = tmp_path / "adj_real"
    rc = main(
        [
            "agentdojo",
            "--suite",
            "workspace",
            "--real",
            "--provider",
            "mock",
            "--defense",
            "sentinel",
            "--limit",
            "4",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["real"] is True
    assert metrics["n"] == 4
    assert metrics["n_benign"] >= 1
    assert metrics["n_attack"] >= 1
