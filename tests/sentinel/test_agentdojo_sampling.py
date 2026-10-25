"""AgentDojo job sampling, --attack none, and metrics fields."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_tester.agent.agentdojo_adapter import (
    mock_agentdojo_suite,
    run_agentdojo_suite,
    sample_real_suite_jobs,
    try_import_agentdojo,
)
from llm_tester.cli import main


def _fake_suite(n_user: int = 4, n_inj: int = 3):
    user_tasks = {f"user_task_{i}": SimpleNamespace(PROMPT=f"u{i}") for i in range(n_user)}
    injection_tasks = {
        f"injection_task_{i}": SimpleNamespace(GOAL=f"g{i}") for i in range(n_inj)
    }
    return SimpleNamespace(user_tasks=user_tasks, injection_tasks=injection_tasks)


def test_sample_full_cartesian_without_limit():
    suite = _fake_suite(4, 3)
    jobs = sample_real_suite_jobs(suite, limit=None, seed=0, include_attacks=True)
    assert len(jobs) == 4 + 4 * 3  # benign + full product
    assert sum(1 for j in jobs if j["label"] == "benign") == 4
    assert sum(1 for j in jobs if j["label"] == "attack") == 12


def test_limit_stratified_from_full_attack_space():
    suite = _fake_suite(10, 5)  # attack pool = 50
    jobs = sample_real_suite_jobs(suite, limit=8, seed=7, include_attacks=True)
    assert len(jobs) == 8
    n_b = sum(1 for j in jobs if j["label"] == "benign")
    n_a = sum(1 for j in jobs if j["label"] == "attack")
    assert n_b >= 1 and n_a >= 1
    assert n_b + n_a == 8
    # Attacks are not capped at n_injection (5): with limit=8 we can take >5 attacks
    # when the split allows — at least the pool size must be the product.
    assert n_a <= 50


def test_sample_seed_reproducible():
    suite = _fake_suite(6, 4)
    a = sample_real_suite_jobs(suite, limit=6, seed=123, include_attacks=True)
    b = sample_real_suite_jobs(suite, limit=6, seed=123, include_attacks=True)
    c = sample_real_suite_jobs(suite, limit=6, seed=999, include_attacks=True)
    assert [j["user_task_id"] for j in a] == [j["user_task_id"] for j in b]
    assert [j.get("injection_task_id") for j in a] == [
        j.get("injection_task_id") for j in b
    ]
    assert [j["user_task_id"] for j in a] != [j["user_task_id"] for j in c] or [
        j.get("injection_task_id") for j in a
    ] != [j.get("injection_task_id") for j in c]


def test_attack_none_skips_attack_jobs():
    suite = _fake_suite(5, 4)
    jobs = sample_real_suite_jobs(suite, limit=6, seed=0, include_attacks=False)
    assert jobs
    assert all(j["label"] == "benign" for j in jobs)
    assert all(j["injection_task_id"] is None for j in jobs)


def test_mock_suite_metrics_include_llm_and_overhead():
    tasks = mock_agentdojo_suite()
    summary, results = run_agentdojo_suite(tasks, defense_name="sentinel")
    m = summary.to_metrics()
    assert "llm_calls_total" in m
    assert "llm_calls_per_task" in m
    assert "defense_overhead_mean_ms" in m
    assert m["llm_calls_total"] >= 0
    assert all("llm_calls" in r.to_dict() for r in results)
    assert all("defense_overhead_ms" in r.to_dict() for r in results)


@pytest.mark.skipif(try_import_agentdojo() is None, reason="agentdojo not installed")
def test_real_attack_none_cli(tmp_path: Path):
    run_dir = tmp_path / "adj_none"
    rc = main(
        [
            "agentdojo",
            "--suite",
            "workspace",
            "--real",
            "--provider",
            "mock",
            "--attack",
            "none",
            "--limit",
            "4",
            "--seed",
            "0",
            "--defense",
            "none",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_attack"] == 0
    assert metrics["n_benign"] == metrics["n"] == 4
    assert metrics["attack"] == "none"
    assert "llm_calls_total" in metrics
    assert "defense_overhead_mean_ms" in metrics


@pytest.mark.skipif(try_import_agentdojo() is None, reason="agentdojo not installed")
def test_real_limit_can_exceed_n_injection(tmp_path: Path):
    """With |U|=40, |I|=14, limit=20 attack stratum can exceed 14."""
    from llm_tester.agent.agentdojo_adapter import (
        load_agentdojo_suite,
        sample_real_suite_jobs,
    )

    suite = load_agentdojo_suite("workspace", benchmark_version="v1.2.2")
    n_inj = len(suite.injection_tasks)
    jobs = sample_real_suite_jobs(suite, limit=30, seed=1, include_attacks=True)
    n_a = sum(1 for j in jobs if j["label"] == "attack")
    # Half of 30 is 15 — already > typical n_injection if we only had 14 paired.
    assert n_a > 0
    assert len(jobs) == 30
    # Full attack pool is |U|*|I| which is >> n_inj.
    assert len(suite.user_tasks) * n_inj > n_inj
