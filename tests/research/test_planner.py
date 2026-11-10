"""Planner rotation / category selection (deterministic, offline)."""

from __future__ import annotations


def test_a_rerun_comes_before_rotation(research_runner):
    planner = research_runner["planner"]
    jobs = planner.plan(n=3, dry=True, simulate_done=True)
    assert jobs, "expected at least one planned job"
    assert jobs[0]["category"] == "a_rerun"
    assert jobs[0]["id"].startswith("a-")
    assert all(j["source"] == "planner" for j in jobs)


def test_rotation_picks_f_real_mcp_early_by_weight(research_runner):
    """With a_rerun exhausted, weighted rotation should surface f_real_mcp among the first batch.

    Rotation order in policy: b_seeds(w=3), f_real_mcp(w=1), d_variants(w=2), …
    With zero counts, ties break by list order → b first; after one b job, f (0/1) beats b (1/3).
    """
    planner = research_runner["planner"]
    # Exhaust a_rerun by simulating the full static list as already queued.
    existing = {}
    pol = planner.policy()
    while True:
        j = planner.gen_a(existing, pol)
        if not j:
            break
        existing[j["id"]] = "done"

    # First non-a job should be b_seeds (tie-break order).
    j1 = planner.next_job(existing, pol, {})
    assert j1 is not None
    assert j1["category"] == "b_seeds"
    existing[j1["id"]] = "pending"

    # Next: f_real_mcp has count 0 / weight 1 = 0; b has 1/3 ≈ 0.33 → f wins.
    j2 = planner.next_job(existing, pol, {})
    assert j2 is not None
    assert j2["category"] == "f_real_mcp"
    assert j2["kind"] == "real_mcp"
    assert "oberih_gw" in j2["arms"]


def test_plan_dry_does_not_write_queue(research_runner):
    planner = research_runner["planner"]
    common = research_runner["common"]
    before = list(common.Q["pending"].glob("*.json"))
    planner.plan(n=2, dry=True, simulate_done=True)
    after = list(common.Q["pending"].glob("*.json"))
    assert before == after
