"""Runner copy of scripts_v6/v6shim.py: route scenario export to mcp-security-research-v6 (--corpus $V6_CORPUS)
and override n_variants with $V6_NVAR. No repo edits."""
import os, sys, subprocess, tempfile
from pathlib import Path
import llm_tester.agent.import_mcp_sim as IMS
import llm_tester.agent.mcp_scenario_eval as MSE

# RESEARCH_ROOT / OBERIH_TESTS_ROOT = workspace that holds corpus checkouts (sibling of runner/).
ROOT = Path(
    os.environ.get("RESEARCH_ROOT")
    or os.environ.get("OBERIH_TESTS_ROOT")
    or Path(__file__).resolve().parents[2]  # lib/ → runner/ → research workspace
)
V6 = ROOT / "mcp-security-research-v6"
CORPUS = os.environ.get("V6_CORPUS", "v6-hard")

def _export(research_root, *, lang="en", scenario_ids=None):
    tmp = tempfile.NamedTemporaryFile(prefix="mcp-v6-export-", suffix=".jsonl", delete=False); tmp.close()
    cmd = [sys.executable, "-m", "scenarios.export", "--out", tmp.name, "--lang", lang, "--corpus", CORPUS]
    for sid in scenario_ids or []: cmd += ["--id", sid]
    p = subprocess.run(cmd, cwd=str(V6), capture_output=True, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr or p.stdout)
    return Path(tmp.name)
IMS.export_from_research_root = _export
_orig = MSE.load_scenario_traces
def _lst(*a, **kw):
    kw["n_variants"] = int(os.environ.get("V6_NVAR", "2"))
    return _orig(*a, **kw)
MSE.load_scenario_traces = _lst
