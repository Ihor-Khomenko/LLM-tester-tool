from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_tester.datasets import PromptItem
from llm_tester.providers import ChatResult, LLMProvider
from llm_tester.runner import load_completed_ids, run_evaluation


def _provider(responses):
    provider = MagicMock()
    provider.info.return_value = {
        "provider": "mock",
        "model": "m",
        "base_url": "http://example",
        "params": {"temperature": 0.0},
    }
    queue = list(responses)

    def chat(prompt, **kwargs):
        if not queue:
            raise AssertionError("unexpected extra chat call")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(content=item, latency_s=0.01)

    provider.chat.side_effect = chat
    return provider


def test_load_completed_ids(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text(
        '{"id":"a","status":"refusal"}\n{"id":"b","status":"error"}\n',
        encoding="utf-8",
    )
    assert load_completed_ids(path) == {"a", "b"}


def test_resume_skips_done_and_appends(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    out.write_text(
        '{"id":"1","status":"refusal","refused":true}\n',
        encoding="utf-8",
    )
    prompts = [
        PromptItem(id="1", prompt="already done", dataset="advbench"),
        PromptItem(id="2", prompt="new one", dataset="advbench"),
    ]
    provider = _provider(["I cannot help with that."])
    summary = run_evaluation(
        prompts,
        provider,
        out_path=out,
        resume=True,
        verbose=False,
    )
    assert summary.resumed_skipped == 1
    assert summary.requests == 1
    assert summary.refusals == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"id": "2"' in lines[1]


def test_max_requests_stops_early(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    prompts = [
        PromptItem(id=str(i), prompt=f"p{i}", dataset="advbench") for i in range(5)
    ]
    provider = _provider(["ok"] * 5)
    summary = run_evaluation(
        prompts,
        provider,
        out_path=out,
        max_requests=2,
        verbose=False,
    )
    assert summary.requests == 2
    assert summary.stopped_max_requests is True
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_delay_between_requests(tmp_path: Path):
    out = tmp_path / "out.jsonl"
    prompts = [
        PromptItem(id="1", prompt="a", dataset="x"),
        PromptItem(id="2", prompt="b", dataset="x"),
    ]
    provider = _provider(["I cannot help.", "I cannot help."])
    with patch("llm_tester.runner.time.sleep") as sleep:
        run_evaluation(
            prompts,
            provider,
            out_path=out,
            delay=1.5,
            verbose=False,
        )
    sleep.assert_called_once_with(1.5)


def test_429_retries_with_backoff():
    from openai import RateLimitError

    mock_client = MagicMock()
    ok_choice = MagicMock()
    ok_choice.message.content = "I cannot help."
    ok_resp = MagicMock()
    ok_resp.choices = [ok_choice]
    ok_resp.model_dump.return_value = {}

    err = RateLimitError(
        message="rate limited",
        response=MagicMock(status_code=429, headers={"retry-after": "0"}),
        body=None,
    )
    mock_client.chat.completions.create.side_effect = [err, ok_resp]

    with patch("llm_tester.providers.OpenAI", return_value=mock_client):
        with patch("llm_tester.providers.time.sleep") as sleep:
            provider = LLMProvider("ollama", "llama3", max_retries=3, backoff_base=0.01)
            result = provider.chat("hi")

    assert result.content == "I cannot help."
    assert result.attempts == 2
    assert sleep.called
