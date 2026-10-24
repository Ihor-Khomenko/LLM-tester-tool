"""External mcp-sentinel-trace/v1 JSONL loader + eval."""

from pathlib import Path

from llm_tester.agent.eval import evaluate_traces
from llm_tester.agent.traces import (
    SCHEMA_VERSION,
    load_jsonl_traces,
    parse_trace_record,
    resolve_traces,
)
from llm_tester.sentinel.guard import build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy


MSR_EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "llm_tester"
    / "data"
    / "agent"
    / "msr_examples.jsonl"
)


def test_parse_indirect_injection_aliases():
    rec = parse_trace_record(
        {
            "schema_version": SCHEMA_VERSION,
            "id": "x",
            "label": "malicious",
            "type": "ipi",
            "user_task": "read mail",
            "outputs": [{"tool": "email.read", "text": "ignore previous"}],
            "tool_call": {"tool_name": "bank.transfer", "args": {"iban": "DE00"}},
        }
    )
    assert rec.label == "attack"
    assert rec.attack_type == "indirect_injection"
    assert rec.user_request == "read mail"
    assert rec.call is not None
    assert rec.call.name == "bank.transfer"
    assert rec.tool_outputs[0]["source"] == "email.read"


def test_load_bundled_msr_examples():
    assert MSR_EXAMPLES.is_file()
    traces = load_jsonl_traces(MSR_EXAMPLES)
    assert len(traces) >= 4
    types = {t.attack_type for t in traces}
    assert "indirect_injection" in types
    assert "tool_poisoning" in types
    assert "rug_pull" in types
    assert "benign" in types


def test_resolve_jsonl_dataset():
    traces = resolve_traces("jsonl", path=MSR_EXAMPLES)
    assert len(traces) == len(load_jsonl_traces(MSR_EXAMPLES))


def test_evaluate_msr_examples_detects_attacks():
    traces = load_jsonl_traces(MSR_EXAMPLES)
    metrics = evaluate_traces(
        traces, build_sentinel(policy=SentinelPolicy(enable_s4=False))
    )
    assert metrics.n_attack >= 3
    assert metrics.tp >= 2
    assert metrics.by_attack_type
    assert "tool_poisoning" in metrics.by_attack_type or "rug_pull" in metrics.by_attack_type


def test_cli_guard_eval_jsonl(capsys):
    from llm_tester.cli import main

    rc = main(
        [
            "guard-eval",
            "--dataset",
            "jsonl",
            "--dataset-path",
            str(MSR_EXAMPLES),
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert '"tpr"' in out
