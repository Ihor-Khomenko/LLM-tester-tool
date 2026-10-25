"""Grey-zone rebuild + guard-eval --repeats (offline mocks)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_tester.agent.eval import evaluate_traces
from llm_tester.agent.grey_zone import rebuild_grey_zone_subset, select_grey_zone_ids
from llm_tester.agent.repeats import aggregate_repeat_metrics
from llm_tester.agent.traces import load_synthetic_traces
from llm_tester.artifacts.run_dir import finalize_run_dir, open_results_writer
from llm_tester.cli import main
from llm_tester.sentinel.guard import build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy


def test_select_grey_zone_ids_escalated():
    records = [
        {"id": "a", "escalated": True, "reasons": ["s4:block"]},
        {"id": "b", "escalated": False, "reasons": []},
        {
            "id": "c",
            "escalated": False,
            "decisions": [
                {
                    "stage": "cascade",
                    "stage_scores": [{"stage": "S4", "score": 0.9, "details": {}}],
                }
            ],
        },
    ]
    assert select_grey_zone_ids(records, filter_mode="escalated") == ["a"]
    assert select_grey_zone_ids(records, filter_mode="s4-stage") == ["c"]
    assert select_grey_zone_ids(records, filter_mode="both") == ["a", "c"]


def test_rebuild_grey_zone_subset_cli(tmp_path: Path):
    traces = load_synthetic_traces()
    policy = SentinelPolicy(enable_s4=False, enable_s3=True, s3_backend="lite")
    guard = build_sentinel(policy=policy)
    metrics, records = evaluate_traces(traces, guard, collect_records=True)
    # Force a couple of escalated flags for the filter (offline without S4).
    for i, rec in enumerate(records):
        if i % 3 == 0:
            rec["escalated"] = True

    run_dir = tmp_path / "src_run"
    with open_results_writer(run_dir, append=False) as w:
        for rec in records:
            w.write(rec)
    finalize_run_dir(
        run_dir,
        command="guard-eval",
        config={"command": "guard-eval", "dataset": "synthetic"},
        metrics=metrics.to_dict(),
        argv=["guard-eval", "--dataset", "synthetic"],
        provider=None,
        model=None,
    )

    out = tmp_path / "grey.jsonl"
    rc = main(
        [
            "grey-zone-rebuild",
            "--from-run",
            str(run_dir),
            "--dataset",
            "synthetic",
            "--filter",
            "escalated",
            "--out",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    assert out.is_file()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == sum(1 for r in records if r.get("escalated"))
    man = json.loads((tmp_path / "grey_manifest.json").read_text(encoding="utf-8"))
    assert man["n"] == len(lines)
    assert man["filter"] == "escalated"


def test_aggregate_repeat_metrics_mean_std():
    per = [
        {"tpr": 0.5, "fpr": 0.1, "asr_with_guard": 0.5},
        {"tpr": 1.0, "fpr": 0.0, "asr_with_guard": 0.0},
    ]
    agg = aggregate_repeat_metrics(per)
    assert agg["tpr"]["n"] == 2
    assert abs(agg["tpr"]["mean"] - 0.75) < 1e-9
    assert agg["tpr"]["std"] is not None and agg["tpr"]["std"] > 0
    assert abs(agg["fpr"]["mean"] - 0.05) < 1e-9


def test_guard_eval_repeats_cli(tmp_path: Path):
    run_dir = tmp_path / "ge_rep"
    rc = main(
        [
            "guard-eval",
            "--dataset",
            "synthetic",
            "--repeats",
            "3",
            "--repeat-seed-stride",
            "1",
            "--seed",
            "0",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc == 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["repeats"] == 3
    assert len(metrics["per_repeat"]) == 3
    assert "aggregate" in metrics
    assert metrics["aggregate"]["tpr"]["n"] == 3
    assert (run_dir / "repeat_0" / "results.jsonl.gz").is_file()
