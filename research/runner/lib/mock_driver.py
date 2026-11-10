"""Mock driver for runner self-tests: no Ollama, no LLM. Writes fake result rows marked mock=true."""
import argparse, json, time, datetime
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--model", default="mock")
ap.add_argument("--run-tag", default=""); ap.add_argument("--defenses", default="none,sentinel"); ap.add_argument("--sleep", type=float, default=1.0)
ap.add_argument("--fail", action="store_true")
a, _ = ap.parse_known_args()
if a.fail: raise SystemExit(3)
for i, lab in enumerate(["attack", "benign"]):
    time.sleep(a.sleep)
    res = {d: {"defense": "oberih" if d == "sentinel" else d, "status": "scored", "attack_success_strict": False, "attack_success_stock": False,
               "utility_success": True if lab == "benign" else None, "blocked": False, "executed_calls": [], "latency_ms": 1.0,
               "t0": [], "guard_steps": []} for d in a.defenses.split(",")}
    e = {"model": a.model, "lang": "en", "id": f"mock-{lab}-{i}", "base_id": f"mock-{lab}", "label": lab, "family": "mock",
         "case_status": "scored", "mock": True, "run_tag": a.run_tag, "ts": datetime.datetime.now().isoformat(), "results": res}
    with open(a.out, "a") as f: f.write(json.dumps(e) + "\n")
print("DONE 2")
