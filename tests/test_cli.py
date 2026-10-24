from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_tester.cli import main
from llm_tester.providers import ChatResult


def test_cli_report(tmp_path: Path, capsys):
    results = tmp_path / "r.jsonl"
    results.write_text(
        '{"status":"refusal","refused":true,"attack_success":false,'
        '"provider":"ollama","model":"llama3","latency_s":0.2}\n',
        encoding="utf-8",
    )
    rc = main(["report", str(results)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Refusal rate" in out


def test_cli_datasets(capsys):
    rc = main(["datasets"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "advbench" in out
    assert "promptinject" in out
    assert "jailbreakbench" in out


def test_cli_run_with_mocks(tmp_path: Path):
    dataset = tmp_path / "hq.json"
    dataset.write_text(
        '[{"id":1,"topic":"t","subtopic":"s","question":"bad?"}]',
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"

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
                "--out",
                str(out),
                "--quiet",
            ]
        )
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "I cannot provide that." in text
    assert '"provider": "ollama"' in text
