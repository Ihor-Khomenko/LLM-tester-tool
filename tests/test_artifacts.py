"""Offline tests for run artefacts, secrets redaction, index, compare."""

from __future__ import annotations

import json
from pathlib import Path

from llm_tester.artifacts.compare import compare_runs, format_compare
from llm_tester.artifacts.env_info import collect_env
from llm_tester.artifacts.index import build_index, write_index_csv
from llm_tester.artifacts.run_dir import (
    finalize_run_dir,
    load_metrics,
    load_results,
    open_results_writer,
)
from llm_tester.artifacts.secrets import redact_mapping, scrub_secrets
from llm_tester.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "runs"


def test_scrub_secrets_redacts_api_keys():
    payload = {
        "provider": "openai",
        "api_key": "sk-secret-value",
        "nested": {"token": "abc", "model": "gpt"},
        "OPENAI_API_KEY": "sk-x",
    }
    clean = scrub_secrets(payload)
    assert clean["api_key"] == "***REDACTED***"
    assert clean["nested"]["token"] == "***REDACTED***"
    assert clean["nested"]["model"] == "gpt"
    assert clean["OPENAI_API_KEY"] == "***REDACTED***"


def test_redact_mapping_keeps_non_secrets():
    data = redact_mapping({"temperature": 0.0, "seed": 42, "model": "llama3"})
    assert data["temperature"] == 0.0
    assert data["model"] == "llama3"


def test_collect_env_has_no_secret_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-appear")
    env = collect_env(provider="openai", model="gpt-4o-mini", command="run")
    dumped = json.dumps(env)
    assert "sk-should-not-appear" not in dumped
    assert env["credential_env_flags"].get("OPENAI_API_KEY") == "set"
    assert "timestamp_utc" in env
    assert "hardware" in env


def test_finalize_run_dir_writes_all_files(tmp_path: Path):
    run_dir = tmp_path / "runs" / "20260101_demo"
    with open_results_writer(run_dir) as w:
        w.write({"id": "1", "attack_success": False, "latency_s": 0.2, "api_key": "nope"})
    bundle = finalize_run_dir(
        run_dir,
        command="run",
        config={"command": "run", "api_key": "secret", "dataset": "advbench"},
        metrics={"n": 1, "attack_success_rate": 0.0},
        argv=["run", "--api-key", "secret", "--dataset", "advbench"],
        provider="ollama",
        model="llama3",
        extra_sha=["mcp-security-research=deadbeef"],
    )
    assert bundle.config_path.exists()
    assert bundle.env_path.exists()
    assert bundle.metrics_path.exists()
    assert bundle.readme_path.exists()
    assert bundle.results_path.exists()
    cfg_text = bundle.config_path.read_text(encoding="utf-8")
    assert "secret" not in cfg_text or "***REDACTED***" in cfg_text
    rows = load_results(run_dir)
    assert rows[0]["api_key"] == "***REDACTED***"
    env = json.loads(bundle.env_path.read_text(encoding="utf-8"))
    assert env["extra_sha"] == ["mcp-security-research=deadbeef"]


def test_index_and_compare_fixtures(tmp_path: Path):
    rows = build_index(FIXTURES)
    assert len(rows) >= 3
    out = tmp_path / "index.csv"
    write_index_csv(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "run_id" in text
    assert "tpr" in text

    comparison = compare_runs([FIXTURES / "run_a", FIXTURES / "run_b"])
    md = format_compare(comparison, fmt="md")
    assert "tpr" in md
    csv = format_compare(comparison, fmt="csv")
    assert "metric" in csv
    docx = format_compare(comparison, fmt="docx-ready", table_number=2)
    assert "Таблиця 4.2" in docx
    assert docx.index("Таблиця") < docx.index("|")


def test_cli_index_compare(tmp_path: Path, capsys):
    rc = main(["index", str(FIXTURES), "--out", str(tmp_path / "idx.csv")])
    assert rc == 0
    assert (tmp_path / "idx.csv").exists()

    rc = main(
        [
            "compare",
            str(FIXTURES / "run_a"),
            str(FIXTURES / "run_b"),
            "--format",
            "docx-ready",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Таблиця 4.1" in out


def test_cli_guard_eval_writes_run_dir(tmp_path: Path):
    run_dir = tmp_path / "ge_run"
    rc = main(
        [
            "guard-eval",
            "--dataset",
            "synthetic",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 0
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "results.jsonl.gz").exists()
    assert (run_dir / "README.md").exists()
    metrics = load_metrics(run_dir)
    assert "tpr" in metrics
    assert metrics["n"] > 0
    # no secrets in env
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert "api_key" not in json.dumps(env).lower() or "REDACTED" in json.dumps(env)


def test_cli_run_writes_run_dir(tmp_path: Path):
    dataset = tmp_path / "hq.json"
    dataset.write_text(
        '[{"id":1,"topic":"t","subtopic":"s","question":"bad?"}]',
        encoding="utf-8",
    )
    run_dir = tmp_path / "run_out"
    from unittest.mock import MagicMock, patch

    from llm_tester.providers import ChatResult

    mock_provider = MagicMock()
    mock_provider.info.return_value = {
        "provider": "ollama",
        "model": "llama3",
        "base_url": "http://localhost:11434/v1",
        "params": {"temperature": 0.0, "seed": 42},
    }
    mock_provider.chat.return_value = ChatResult(
        content="I cannot provide that.", latency_s=0.1
    )
    with patch("llm_tester.cli.LLMProvider", return_value=mock_provider):
        rc = main(
            [
                "run",
                "--dataset",
                "harmfulqa",
                "--dataset-path",
                str(dataset),
                "--provider",
                "ollama",
                "--model",
                "llama3",
                "--range",
                "1",
                "--run-dir",
                str(run_dir),
                "--quiet",
            ]
        )
    assert rc == 0
    assert (run_dir / "results.jsonl.gz").exists()
    assert (run_dir / "metrics.json").exists()
    rows = load_results(run_dir)
    assert rows[0]["refused"] is True
