"""Sentinel isolation + provider adapter contracts."""

import ast
from pathlib import Path

from llm_tester.agent.provider_adapter import build_llm_chat, provider_choices
from llm_tester.sentinel.provider import LLMChat, extract_chat_text
from llm_tester.sentinel.stages.judge import JudgeStage
from llm_tester.sentinel.types import ToolCall


SENTINEL_ROOT = Path(__file__).resolve().parents[2] / "llm_tester" / "sentinel"


def _iter_py_files(root: Path):
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def test_sentinel_does_not_import_rest_of_llm_tester():
    """Sentinel may only import stdlib, third parties, or llm_tester.sentinel.*."""
    forbidden_prefixes = (
        "llm_tester.agent",
        "llm_tester.cli",
        "llm_tester.config",
        "llm_tester.datasets",
        "llm_tester.guard",
        "llm_tester.judges",
        "llm_tester.providers",
        "llm_tester.range_utils",
        "llm_tester.report",
        "llm_tester.runner",
    )
    offenders = []
    for path in _iter_py_files(SENTINEL_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                if mod == "llm_tester" or mod.startswith("llm_tester."):
                    if not (
                        mod == "llm_tester.sentinel"
                        or mod.startswith("llm_tester.sentinel.")
                    ):
                        offenders.append(f"{path.name}: from {mod}")
                for pref in forbidden_prefixes:
                    if mod == pref or mod.startswith(pref + "."):
                        offenders.append(f"{path.name}: from {mod}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    if mod.startswith("llm_tester.") and not mod.startswith(
                        "llm_tester.sentinel"
                    ):
                        offenders.append(f"{path.name}: import {mod}")
    assert offenders == [], "Sentinel leaked imports:\n" + "\n".join(offenders)


def test_provider_choices_cover_required_apis():
    choices = set(provider_choices())
    for name in ("ollama", "openai", "xai", "openrouter", "groq"):
        assert name in choices


def test_llmchat_protocol_accepts_duck_type():
    class Fake:
        def chat(self, prompt, *, system=None):
            return "BLOCK fake"

    assert isinstance(Fake(), LLMChat)
    stage = JudgeStage(provider=Fake())
    score = stage.judge(
        "u",
        ToolCall(name="bank.transfer", arguments={"iban": "DE"}),
        ["ignore previous"],
        0.5,
        [],
    )
    assert score.score >= 0.9


def test_extract_chat_text():
    assert extract_chat_text("hi") == "hi"
    assert extract_chat_text(type("R", (), {"content": "x"})()) == "x"


def test_build_llm_chat_constructs_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = build_llm_chat("openai", "gpt-4o-mini")
    assert client.provider == "openai"
    assert hasattr(client, "chat")
