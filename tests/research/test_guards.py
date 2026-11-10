"""Supervisor guard logic (disk/swap/battery latch), offline with monkeypatched sensors."""

from __future__ import annotations

import importlib
import sys


def _load_supervisor(research_runner):
    if "supervisor" in sys.modules:
        del sys.modules["supervisor"]
    return importlib.import_module("supervisor")


def test_battery_min_pct_zero_clears_latch(research_runner, monkeypatch):
    supervisor = _load_supervisor(research_runner)
    common = research_runner["common"]
    pol = dict(research_runner["policy"])
    pol["guards"] = dict(pol["guards"])
    pol["guards"]["battery_min_pct"] = 0
    pol["guards"]["require_ollama"] = False

    monkeypatch.setattr(common, "free_gb", lambda: 100.0)
    monkeypatch.setattr(common, "swap_used_mb", lambda: 0.0)
    monkeypatch.setattr(common, "battery", lambda: ("Battery", 10))
    monkeypatch.setattr(common, "ollama_up", lambda: True)
    # supervisor imported free_gb etc via `from common import *` — patch supervisor module too
    monkeypatch.setattr(supervisor, "free_gb", lambda: 100.0)
    monkeypatch.setattr(supervisor, "swap_used_mb", lambda: 0.0)
    monkeypatch.setattr(supervisor, "battery", lambda: ("Battery", 10))
    monkeypatch.setattr(supervisor, "ollama_up", lambda: True)

    supervisor.STATE["battery_latched"] = True
    blockers, env = supervisor.guards(pol, check_legacy=False)
    assert supervisor.STATE["battery_latched"] is False
    assert not any(b.startswith("battery_") for b in blockers)
    assert env["power"] == "Battery"
    assert env["battery_pct"] == 10


def test_battery_latch_engages_when_threshold_positive(research_runner, monkeypatch):
    supervisor = _load_supervisor(research_runner)
    pol = dict(research_runner["policy"])
    pol["guards"] = dict(pol["guards"])
    pol["guards"]["battery_min_pct"] = 30
    pol["guards"]["require_ollama"] = False

    monkeypatch.setattr(supervisor, "free_gb", lambda: 100.0)
    monkeypatch.setattr(supervisor, "swap_used_mb", lambda: 0.0)
    monkeypatch.setattr(supervisor, "battery", lambda: ("Battery", 10))
    monkeypatch.setattr(supervisor, "ollama_up", lambda: True)

    supervisor.STATE["battery_latched"] = False
    blockers, _ = supervisor.guards(pol, check_legacy=False)
    assert supervisor.STATE["battery_latched"] is True
    assert any(b.startswith("battery_") for b in blockers)

    # AC clears the latch
    monkeypatch.setattr(supervisor, "battery", lambda: ("AC", 10))
    blockers, _ = supervisor.guards(pol, check_legacy=False)
    assert supervisor.STATE["battery_latched"] is False
    assert not any(b.startswith("battery_") for b in blockers)


def test_disk_guard_blocks(research_runner, monkeypatch):
    supervisor = _load_supervisor(research_runner)
    pol = dict(research_runner["policy"])
    pol["guards"] = dict(pol["guards"])
    pol["guards"]["min_free_gb"] = 50
    pol["guards"]["battery_min_pct"] = 0
    pol["guards"]["require_ollama"] = False

    monkeypatch.setattr(supervisor, "free_gb", lambda: 5.0)
    monkeypatch.setattr(supervisor, "swap_used_mb", lambda: 0.0)
    monkeypatch.setattr(supervisor, "battery", lambda: ("AC", 100))
    monkeypatch.setattr(supervisor, "ollama_up", lambda: True)

    blockers, env = supervisor.guards(pol, check_legacy=False)
    assert any(b.startswith("disk_free_") for b in blockers)
    assert env["free_gb"] == 5.0
