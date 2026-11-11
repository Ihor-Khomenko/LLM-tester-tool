"""Planner ETA / max_job_min arm splitting."""

from __future__ import annotations


def test_est_minutes_scales_with_n_variants(research_runner):
    planner = research_runner["planner"]
    j2 = planner.job(
        "t-nvar2", "manual", model="granite3-moe:3b", corpus="v6-hard",
        arms=["none", "oberih"], n_variants=2, temperature=0.7, seed=1,
    )
    j4 = planner.job(
        "t-nvar4", "manual", model="granite3-moe:3b", corpus="v6-hard",
        arms=["none", "oberih"], n_variants=4, temperature=0.7, seed=1,
    )
    assert j4["est_min"] > j2["est_min"]
    assert abs(j4["est_min"] / j2["est_min"] - 2.0) < 0.25


def test_split_job_by_arms_adds_p2_p3(research_runner, monkeypatch):
    planner = research_runner["planner"]
    pol = research_runner["policy"]
    pol = dict(pol)
    pol["max_job_min"] = 30
    # Force high ETA by using many arms + slow model seconds.
    pol["sec_per_case_arm"] = dict(pol.get("sec_per_case_arm") or {})
    pol["sec_per_case_arm"]["granite3-moe:3b"] = 60
    monkeypatch.setattr(planner, "policy", lambda: pol)

    j = planner.job(
        "p-att-test",
        "manual",
        model="granite3-moe:3b",
        corpus="v6-hard",
        arms=["none", "oberih", "oberih_graded", "soft_s3", "prompt_guard2"],
        n_variants=2,
        temperature=0.7,
        seed=3101,
        labels=["attack"],
    )
    parts = planner.split_job_by_arms(j, pol)
    assert len(parts) >= 2
    assert parts[0]["id"] == "p-att-test"
    assert parts[1]["id"] == "p-att-test-p2"
    # Pairing: same model/seed/corpus/labels across parts; arms partitioned.
    all_arms = []
    for p in parts:
        assert p["model"] == j["model"]
        assert p["seed"] == j["seed"]
        assert p["corpus"] == j["corpus"]
        assert p["split_of"] == "p-att-test"
        all_arms.extend(p["arms"])
    assert sorted(all_arms) == sorted(j["arms"])
    # none/oberih are ordered first; they share a part only when the pair fits under max_job_min
    assert "none" in all_arms and "oberih" in all_arms
    assert parts[0]["arms"][0] in ("none", "oberih")


def test_split_noop_when_under_budget(research_runner):
    planner = research_runner["planner"]
    j = planner.job(
        "small", "manual", model="granite3-moe:3b", corpus="v6-hard",
        arms=["none", "oberih"], n_variants=2, temperature=0.7, seed=1,
    )
    parts = planner.split_job_by_arms(j)
    assert len(parts) == 1
    assert parts[0]["id"] == "small"
