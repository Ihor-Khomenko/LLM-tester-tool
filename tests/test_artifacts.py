"""Offline tests for reproducible run artifacts, index, and compare."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_tester.artifacts import (
    collect_env,
    compare_metrics,
    finalize_run,
    format_compare_markdown,
    format_compare_table,
    is_run_dir,
    load_run,
    parse_extra_sha,
    redact_secrets,
    resolve_run_dir,
    write_index,
    write_jsonl_gz,
)
from llm_tester.cli import main
from llm_tester.providers import ChatResult


def test_redact_secrets_strips_api_keys():
    raw = {
        "provider": "openai",
        "api_key": "sk-secret-should-not-appear",
        "OPENAI_API_KEY": "sk-abc",
        "nested": {"access_token": "tok", "model": "gpt"},
        "safe": "hello",
    }
    clean = redact_secrets(raw)
    assert clean["api_key"] == "<redacted>"
    assert clean["OPENAI_API_KEY"] == "<redacted>"
    assert clean["nested"]["access_token"] == "<redacted>"
    assert clean["nested"]["model"] == "gpt"
    assert clean["safe"] == "hello"
    assert "sk-secret" not in json.dumps(clean)


def test_resolve_run_dir_default_and_override(tmp_path: Path):
    auto = resolve_run_dir(
        None,
        run_name="demo",
        runs_root=tmp_path,
        command="run",
        dataset="advbench",
        provider="ollama",
        model="llama3",
    )
    assert auto.parent == tmp_path
    assert auto.name.endswith("_demo")
    custom = resolve_run_dir(tmp_path / "custom_run")
    assert custom == tmp_path / "custom_run"


def test_parse_extra_sha_name_equals():
    name, sha = parse_extra_sha("mcp-security-research=deadbeef")
    assert name == "mcp-security-research"
    assert sha == "deadbeef"


def test_parse_extra_sha_invalid():
    with pytest.raises(ValueError):
        parse_extra_sha("not-a-valid-spec-without-equals")


def test_finalize_run_writes_all_files(tmp_path: Path):
    run_dir = tmp_path / "20260101_demo"
    art = finalize_run(
        run_dir,
        config={
            "command": "run",
            "dataset": "advbench",
            "provider": "ollama",
            "api_key": "should-not-persist",
        },
        env=collect_env(command="run", provider="ollama", model="llama3"),
        metrics={"n": 3, "refusal_rate": 0.5, "attack_success_rate": 0.5},
        results_records=[
            {"id": "1", "refused": True, "api_key": "nope"},
            {"id": "2", "refused": False},
        ],
    )
    assert is_run_dir(art.run_dir)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "env.json").exists()
    assert (run_dir / "results.jsonl.gz").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "README.md").exists()

    cfg_text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "should-not-persist" not in cfg_text
    assert "<redacted>" in cfg_text or "api_key" in cfg_text

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert "timestamp" in env
    assert "python" in env
    assert "hardware" in env
    assert env.get("provider") == "ollama"

    with gzip.open(run_dir / "results.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 2
    assert rows[0]["api_key"] == "<redacted>"

    loaded = load_run(run_dir)
    assert loaded.metrics["refusal_rate"] == 0.5


def test_index_and_compare(tmp_path: Path):
    a = tmp_path / "run_a"
    b = tmp_path / "run_b"
    finalize_run(
        a,
        config={"command": "guard-eval", "dataset": "synthetic"},
        env={"timestamp": "t0", "command": "guard-eval", "provider": None, "model": None, "git": {}},
        metrics={"n": 10, "tpr": 0.9, "fpr": 0.1, "f1": 0.85},
        results_records=[],
    )
    finalize_run(
        b,
        config={"command": "guard-eval", "dataset": "synthetic"},
        env={"timestamp": "t1", "command": "guard-eval", "provider": None, "model": None, "git": {}},
        metrics={"n": 10, "tpr": 0.8, "fpr": 0.05, "f1": 0.82},
        results_records=[],
    )
    idx = write_index(tmp_path)
    assert idx.exists()
    with idx.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2

    arts = [load_run(a), load_run(b)]
    diff = compare_metrics(arts)
    tpr_row = next(r for r in diff if r["metric"] == "tpr")
    assert tpr_row["run_0"] == 0.9
    assert tpr_row["run_1"] == 0.8
    assert tpr_row["delta_01"] == pytest.approx(-0.1)

    table = format_compare_table(arts, diff)
    assert "tpr" in table
    md = format_compare_markdown(arts, diff)
    assert md.startswith("|")


def test_cli_guard_eval_writes_run_dir(tmp_path: Path, capsys):
    run_dir = tmp_path / "ge_run"
    rc = main(
        [
            "guard-eval",
            "--dataset",
            "synthetic",
            "--run-dir",
            str(run_dir),
            "--extra-sha",
            "mcp-security-research=abc123",
        ]
    )
    assert rc == 0
    assert is_run_dir(run_dir)
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["git"]["extra"]["mcp-security-research"] == "abc123"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "tpr" in metrics
    with gzip.open(run_dir / "results.jsonl.gz", "rt", encoding="utf-8") as fh:
        n = sum(1 for line in fh if line.strip())
    assert n == metrics["n"]
    out = capsys.readouterr().out
    assert "Run artifacts written" in out


def test_cli_run_writes_run_dir(tmp_path: Path):
    dataset = tmp_path / "hq.json"
    dataset.write_text(
        '[{"id":1,"topic":"t","subtopic":"s","question":"bad?"}]',
        encoding="utf-8",
    )
    run_dir = tmp_path / "run_out"
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
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["refusals"] == 1


def test_cli_agent_bench_writes_run_dir(tmp_path: Path):
    run_dir = tmp_path / "ab_run"
    mock_provider = MagicMock()
    mock_provider.chat.return_value = ChatResult(
        content="Final Answer: ok", latency_s=0.01
    )
    with patch("llm_tester.cli.LLMProvider", return_value=mock_provider):
        # agent-bench uses build_llm_chat → LLMProvider inside provider_adapter
        with patch(
            "llm_tester.agent.provider_adapter.LLMProvider",
            return_value=mock_provider,
        ):
            rc = main(
                [
                    "agent-bench",
                    "--provider",
                    "ollama",
                    "--model",
                    "llama3",
                    "--without-guard",
                    "--run-dir",
                    str(run_dir),
                    "--max-requests",
                    "2",
                ]
            )
    assert rc == 0
    assert is_run_dir(run_dir)
    assert (run_dir / "results.jsonl.gz").exists()


def test_cli_index_and_compare(tmp_path: Path, capsys):
    a = tmp_path / "a"
    b = tmp_path / "b"
    finalize_run(
        a,
        config={"command": "run", "dataset": "x"},
        env={"timestamp": "t", "git": {}, "provider": "ollama", "model": "m"},
        metrics={"refusal_rate": 0.4, "attack_success_rate": 0.6, "n": 5},
        results_records=[],
    )
    finalize_run(
        b,
        config={"command": "run", "dataset": "x"},
        env={"timestamp": "t", "git": {}, "provider": "ollama", "model": "m"},
        metrics={"refusal_rate": 0.7, "attack_success_rate": 0.3, "n": 5},
        results_records=[],
    )
    md = tmp_path / "diff.md"
    csv_path = tmp_path / "diff.csv"
    rc = main(["index", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "index.csv").exists()
    rc = main(
        [
            "compare",
            str(a),
            str(b),
            "--markdown",
            str(md),
            "--csv",
            str(csv_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "refusal_rate" in out
    assert md.exists()
    assert csv_path.exists()


def test_report_reads_gzipped_results(tmp_path: Path, capsys):
    path = tmp_path / "results.jsonl.gz"
    write_jsonl_gz(
        path,
        [
            {
                "status": "refusal",
                "refused": True,
                "attack_success": False,
                "provider": "ollama",
                "model": "llama3",
                "latency_s": 0.2,
            }
        ],
    )
    rc = main(["report", str(path)])
    assert rc == 0
    assert "Refusal rate" in capsys.readouterr().out


def test_help_lists_index_compare():
    from llm_tester.cli import build_parser

    help_txt = build_parser().format_help()
    assert "index" in help_txt
    assert "compare" in help_txt
