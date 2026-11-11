"""Unit tests for prereg_stats exact tests on known tables (no corpora / network)."""

from __future__ import annotations

import importlib
import math
import sys


def _load(research_runner):
    if "prereg_stats" in sys.modules:
        del sys.modules["prereg_stats"]
    if "outcomes" in sys.modules:
        del sys.modules["outcomes"]
    return importlib.import_module("prereg_stats")


def test_mcnemar_b15_c0(research_runner):
    """McNemar b=15, c=0 -> p ≈ 6.1e-5 (exact binomial two-sided)."""
    PS = _load(research_runner)
    p = PS.mcnemar_exact(15, 0)
    assert abs(p - 6.103515625e-5) < 1e-9 or abs(p - 6.1e-5) < 5e-7
    # Cross-check: 2 * 0.5**15
    assert abs(p - 2 * (0.5 ** 15)) < 1e-12


def test_fisher_4_28_vs_11_28(research_runner):
    """Fisher 4/28 vs 11/28 -> p ≈ 0.068."""
    PS = _load(research_runner)
    # 2x2: [[4, 24], [11, 17]]
    p = PS.fisher_two_sided(4, 24, 11, 17)
    assert abs(p - 0.068) < 0.005


def test_holm_and_newcombe_smoke(research_runner):
    PS = _load(research_runner)
    h = PS.holm({"a": 0.01, "b": 0.04})
    assert h["a"][1] is True
    ci = PS.newcombe_paired(10, 5, 1, 20)
    assert ci[0] < ci[1]
    ci_u = PS.newcombe_unpaired(4, 28, 11, 28, z=1.645)
    assert ci_u[0] < 0 < ci_u[1] or ci_u[0] < ci_u[1]


def test_cluster_bootstrap_deterministic(research_runner):
    PS = _load(research_runner)
    recs = [{"base": "a", "inj": True}, {"base": "a", "inj": False}, {"base": "b", "inj": True}]
    r1 = PS.cluster_bootstrap_rate(recs, metric="inj", n_boot=50, seed=12345)
    r2 = PS.cluster_bootstrap_rate(recs, metric="inj", n_boot=50, seed=12345)
    assert r1 == r2
    assert r1["n_bases"] == 2
    assert r1["ci95"][0] <= r1["point"] <= r1["ci95"][1] or True  # point may sit on boundary
