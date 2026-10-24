from pathlib import Path
from unittest.mock import MagicMock

from llm_tester.datasets import PromptItem
from llm_tester.guard import NoGuard
from llm_tester.judges import PhraseJudge
from llm_tester.providers import ChatResult
from llm_tester.runner import run_evaluation


def _mock_provider(responses):
    provider = MagicMock()
    provider.info.return_value = {
        "provider": "mock",
        "model": "mock-model",
        "base_url": "http://example",
        "params": {"temperature": 0.0, "seed": 42},
    }
    queue = list(responses)

    def chat(prompt, **kwargs):
        content = queue.pop(0) if queue else ""
        if isinstance(content, Exception):
            raise content
        return ChatResult(content=content, latency_s=0.12)

    provider.chat.side_effect = chat
    return provider


def test_run_logs_every_response(tmp_path: Path):
    prompts = [
        PromptItem(id="1", prompt="bad?", dataset="harmfulqa"),
        PromptItem(id="2", prompt="worse?", dataset="harmfulqa"),
    ]
    provider = _mock_provider(
        [
            "I cannot help with that.",
            "Sure, here is a detailed guide...",
        ]
    )
    out = tmp_path / "out.jsonl"
    summary = run_evaluation(
        prompts,
        provider,
        judge=PhraseJudge(),
        guard=NoGuard(),
        out_path=out,
        verbose=False,
    )
    assert summary.total == 2
    assert summary.refusals == 1
    assert summary.attack_successes == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "provider" in lines[0]
    assert "timestamp" in lines[0]
    assert "params" in lines[0]


def test_run_records_errors(tmp_path: Path):
    prompts = [PromptItem(id="1", prompt="x", dataset="harmfulqa")]
    provider = _mock_provider([RuntimeError("network down")])
    out = tmp_path / "out.jsonl"
    summary = run_evaluation(
        prompts, provider, out_path=out, verbose=False
    )
    assert summary.errors == 1
    assert "error" in out.read_text(encoding="utf-8")
