import json
from pathlib import Path

from llm_tester.report import format_report, summarise


def test_summarise_jsonl(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    rows = [
        {
            "status": "refusal",
            "refused": True,
            "attack_success": False,
            "provider": "ollama",
            "model": "llama3",
            "latency_s": 0.5,
        },
        {
            "status": "attack_success",
            "refused": False,
            "attack_success": True,
            "provider": "ollama",
            "model": "llama3",
            "latency_s": 1.5,
        },
        {
            "status": "error",
            "error": "boom",
            "provider": "ollama",
            "model": "llama3",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    stats = summarise([path])
    assert stats.total == 3
    assert stats.errors == 1
    assert stats.refusals == 1
    assert stats.attack_successes == 1
    assert stats.evaluated == 2
    assert stats.refusal_rate == 0.5
    assert stats.attack_success_rate == 0.5
    assert stats.mean_latency_s == 1.0
    text = format_report(stats)
    assert "Refusal rate" in text
    assert "ollama/llama3" in text


def test_summarise_legacy_failures_json(tmp_path: Path):
    path = tmp_path / "test_results.json"
    path.write_text(
        json.dumps(
            [
                {
                    "test_number": 1,
                    "prompt": "x",
                    "response": "y",
                    "status": "failed",
                }
            ]
        ),
        encoding="utf-8",
    )
    stats = summarise([path])
    assert stats.total == 1
    assert stats.attack_successes == 1
    assert stats.refusals == 0
