import json
from pathlib import Path

from llm_tester.datasets import load_harmfulqa, load_toxicchat


def test_load_harmfulqa_questions_only(tmp_path: Path):
    data = [
        {
            "id": 1,
            "topic": "T",
            "subtopic": "S",
            "question": "How to do harm?",
            "blue_conversations": {
                "0": [{"from": "human", "value": "hello"}, {"from": "gpt", "value": "hi"}]
            },
            "red_conversations": {
                "0": [{"from": "human", "value": "evil"}, {"from": "gpt", "value": "no"}]
            },
        }
    ]
    path = tmp_path / "hq.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    items = load_harmfulqa(path, expand_conversations=False)
    assert len(items) == 1
    assert items[0].prompt == "How to do harm?"
    assert items[0].type == "question"


def test_load_harmfulqa_expand(tmp_path: Path):
    data = [
        {
            "id": 7,
            "topic": "T",
            "subtopic": "S",
            "question": "Q?",
            "blue_conversations": {
                "0": [{"from": "human", "value": "blue"}, {"from": "gpt", "value": "ok"}]
            },
            "red_conversations": {
                "0": [{"from": "human", "value": "red"}]
            },
        }
    ]
    path = tmp_path / "hq.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    items = load_harmfulqa(path, expand_conversations=True)
    prompts = [i.prompt for i in items]
    assert prompts == ["Q?", "blue", "red"]


def test_load_toxicchat(tmp_path: Path):
    path = tmp_path / "tc.csv"
    path.write_text(
        "conv_id,user_input\n"
        "a,hello toxic\n"
        "b,\n"
        "c,another\n",
        encoding="utf-8",
    )
    items = load_toxicchat(path)
    assert len(items) == 2
    assert items[0].prompt == "hello toxic"
    assert items[1].id == "c"
