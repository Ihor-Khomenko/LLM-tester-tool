"""Runner wrapper around the frozen live driver (lib/run_mcp_live_ollama.py, copied from oberih-tests/scripts).
Adds: --corpus std|v6-hard, --nvar N, --ids base_id[,base_id...] (scenario filter). Everything else is passed through.
No repo edits; the scoring/Oberih policy logic is untouched (variants only via the driver's own --soft-s3-allow / --policy-json)."""
import os, sys, runpy, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--corpus", default="std"); ap.add_argument("--nvar", type=int, default=0); ap.add_argument("--ids", default="")
a, rest = ap.parse_known_args()
sys.path.insert(0, str(HERE))
if a.corpus in ("v6", "v6-hard"):
    os.environ["V6_CORPUS"] = "v6-hard"; os.environ["V6_NVAR"] = str(a.nvar or 2)
    import v6shim  # noqa: patches export + n_variants
import llm_tester.agent.mcp_scenario_eval as MSE
_orig = MSE.load_scenario_traces
IDS = {x for x in a.ids.split(",") if x}
def _lst(*args, **kw):
    if a.corpus not in ("v6", "v6-hard") and a.nvar:
        kw["n_variants"] = a.nvar
    ts = _orig(*args, **kw)
    if IDS:
        ts = [t for t in ts if ((t.meta or {}).get("base_id") or t.id.split("__")[0]) in IDS]
    return ts
MSE.load_scenario_traces = _lst
drv = str(HERE / "run_mcp_live_ollama.py")
sys.argv = [drv] + rest
runpy.run_path(drv, run_name="__main__")
