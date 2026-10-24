"""CLI tests for guard-eval / agent-bench / sentinel-proxy help."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_tester.cli import main
from llm_tester.providers import ChatResult


def test_guard_eval_synthetic(capsys):
    rc = main(["guard-eval", "--dataset", "synthetic"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TPR=" in out
    assert "preliminary" in out.lower() or "Preliminary" in out or "preliminary" in out


def test_guard_eval_json(capsys):
    rc = main(["guard-eval", "--dataset", "synthetic", "--json", "--no-s3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"tpr"' in out
    assert '"f1"' in out


def test_guard_eval_ablation_flags(capsys):
    rc = main(["guard-eval", "--dataset", "synthetic", "--no-s1", "--no-s2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ablation" in out.lower() or "TPR=" in out


def test_agent_bench_mocked(tmp_path: Path):
    out = tmp_path / "ab.jsonl"
    mock_provider = MagicMock()
    mock_provider.chat.return_value = ChatResult(
        content="Final Answer: ok", latency_s=0.01
    )
    with patch("llm_tester.cli.LLMProvider", return_value=mock_provider):
        rc = main(
            [
                "agent-bench",
                "--provider",
                "ollama",
                "--model",
                "llama3",
                "--without-guard",
                "--out",
                str(out),
                "--max-requests",
                "2",
            ]
        )
    assert rc == 0
    assert out.exists()


def test_sentinel_proxy_requires_command():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["sentinel-proxy"])
    assert exc.value.code != 0


def test_help_lists_new_commands():
    from llm_tester.cli import build_parser

    help_txt = build_parser().format_help()
    assert "guard-eval" in help_txt
    assert "agent-bench" in help_txt
    assert "sentinel-proxy" in help_txt
