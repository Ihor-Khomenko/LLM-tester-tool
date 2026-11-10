"""Scripted-agent real_mcp smoke (Node + local stdio servers; no Ollama / no network APIs)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REAL_MCP = REPO / "research" / "real_mcp"
DRIVER = REPO / "research" / "runner" / "lib" / "real_mcp_driver.py"


def _node_ok() -> bool:
    return shutil.which("node") is not None or Path(os.environ.get("REAL_MCP_NODE") or "").exists()


def _modules_ok() -> bool:
    return (REAL_MCP / "node_modules" / "@modelcontextprotocol" / "server-filesystem").is_dir()


pytestmark = [
    pytest.mark.skipif(not _node_ok(), reason="Node.js not available"),
    pytest.mark.skipif(not _modules_ok(), reason="research/real_mcp node_modules missing (run npm ci)"),
]


def test_real_mcp_mock_agent_smoke(tmp_path, monkeypatch):
    out = tmp_path / "results.jsonl"
    sandbox = tmp_path / "sandbox"
    env = os.environ.copy()
    env["OBERIH_TESTS_ROOT"] = str(REPO / "research")
    env["RESEARCH_ROOT"] = str(REPO / "research")
    env["REAL_MCP_ROOT"] = str(REAL_MCP)
    env["LLM_TESTER_ROOT"] = str(REPO)
    env["PYTHONPATH"] = str(REPO)
    env["OBERIH_PROXY_CONFIG"] = str(REPO / "llm_tester" / "data" / "sentinel" / "proxy.default.yaml")
    # Strip provider keys so nothing accidental leaks into child envs.
    for k in list(env):
        if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            env.pop(k, None)

    cmd = [
        sys.executable,
        str(DRIVER),
        "--model",
        "mock",
        "--llm",
        "mock",
        "--langs",
        "en",
        "--arms",
        "none,oberih",
        "--ids",
        "rmcp-ipi-email-exfil,rmcp-ben-email",
        "--run-tag",
        "ci-smoke",
        "--out",
        str(out),
        "--sandbox-root",
        str(sandbox),
        "--protocol",
        "native",
    ]
    # Allow subprocess stdio MCP; network sockets remain blocked by global conftest.
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=180)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert out.is_file()
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    by_id = {r["id"]: r for r in rows}

    attack = by_id["rmcp-ipi-email-exfil__en"]
    assert attack["results"]["none"]["status"] == "ok"
    assert attack["results"]["none"]["attack_success"] is True
    # Oberih should block the canary exfil for this scripted attack (same as validation mock1).
    assert attack["results"]["oberih"]["status"] == "ok"
    assert attack["results"]["oberih"]["attack_success"] is False
    assert attack["results"]["oberih"]["n_blocked"] >= 1

    benign = by_id["rmcp-ben-email__en"]
    assert benign["results"]["none"]["status"] == "ok"
    assert benign["results"]["none"]["utility"] is True
    assert benign["results"]["oberih"]["status"] == "ok"
