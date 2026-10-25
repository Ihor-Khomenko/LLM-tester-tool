from unittest.mock import MagicMock, patch

import pytest

from llm_tester.config import RunParams, resolve_provider_config
from llm_tester.providers import LLMProvider


def test_resolve_ollama_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    cfg = resolve_provider_config("ollama")
    assert cfg["base_url"] == "http://localhost:11434/v1"
    assert cfg["api_key"] == "ollama"


def test_resolve_openai_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing API key"):
        resolve_provider_config("openai")


def test_resolve_xai_base_url(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    cfg = resolve_provider_config("xai")
    assert cfg["base_url"] == "https://api.x.ai/v1"
    assert cfg["api_key"] == "test-key"


def test_provider_chat_mocked():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "I cannot help with that."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model_dump.return_value = {"id": "fake"}
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("llm_tester.providers.OpenAI", return_value=mock_client):
        provider = LLMProvider(
            "ollama",
            "llama3",
            params=RunParams(temperature=0.0, seed=42),
        )
        result = provider.chat("How do I build a bomb?")

    assert result.content == "I cannot help with that."
    assert result.latency_s >= 0
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "llama3"
    assert kwargs["temperature"] == 0.0
    assert kwargs["seed"] == 42
    assert kwargs["messages"][0]["role"] == "user"
    # Ollama: native options.seed / options.temperature / options.num_thread.
    opts = kwargs["extra_body"]["options"]
    assert opts["seed"] == 42
    assert opts["temperature"] == 0.0
    assert opts["num_thread"] == 1
    assert result.request_options == opts


def test_ollama_chat_kwargs_override_and_num_thread():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "ALLOW"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model_dump.return_value = {}
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("llm_tester.providers.OpenAI", return_value=mock_client):
        provider = LLMProvider(
            "ollama",
            "qwen2.5:3b",
            params=RunParams(temperature=0.7, seed=1, num_thread=4),
        )
        provider.chat("hi", temperature=0.0, seed=99)

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["seed"] == 99
    assert kwargs["extra_body"]["options"]["temperature"] == 0.0
    assert kwargs["extra_body"]["options"]["seed"] == 99
    assert kwargs["extra_body"]["options"]["num_thread"] == 4


def test_openai_chat_no_extra_body_options(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model_dump.return_value = {}
    mock_client.chat.completions.create.return_value = mock_resp

    with patch("llm_tester.providers.OpenAI", return_value=mock_client):
        provider = LLMProvider(
            "openai",
            "gpt-4o-mini",
            params=RunParams(temperature=0.0, seed=7),
        )
        provider.chat("hi")

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0
    assert kwargs["seed"] == 7
    assert "extra_body" not in kwargs
