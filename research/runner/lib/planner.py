"""Deterministic auto-planner. Called by the supervisor when queue/pending is empty.
Order: (a) PR #29 rug-pull/alias re-runs (static list, policy order) -> then weighted rotation b/d/c/e:
next category = argmin(#jobs_of_category / weight), ties by rotation order. Inside a category the cell with the
widest Wilson 95% CI (from runner/summary.json; unseen cell = width 1.0) / fewest jobs wins; ties by policy list order.
Same inputs (policy.yaml + queue dirs + summary.json) => same plan. `python planner.py --dry N` previews N jobs."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa

def _ids_in_queue():
    return all_job_ids()

def _cells():
    s = read_json(RUNNER / "summary.json", {}) or {}
    return s.get("cells", {})

def _width(cells, model, corpus, temp, arm, metric):
    """Wilson CI width; cell keys may include ``|series|nvarN`` suffixes (analyzer)."""
    t = str(float(temp))
    matches = []
    for key, c in (cells or {}).items():
        parts = str(key).split("|")
        if len(parts) >= 4 and parts[0] == model and parts[1] == corpus and parts[2] == t and parts[3] == arm:
            matches.append(c)
    if not matches:
        c = cells.get(f"{model}|{corpus}|{temp}|{arm}") or cells.get(f"{model}|{corpus}|{t}|{arm}")
        if c:
            matches = [c]
    if not matches:
        return 1.0
    if metric == "inj":
        k, n = sum(c.get("inj", 0) for c in matches), sum(c.get("n_att", 0) for c in matches)
    else:
        k, n = sum(c.get("fp", 0) for c in matches), sum(c.get("n_ben", 0) for c in matches)
    lo, hi = wilson(k, n); return hi - lo

def job(jid, category, kind="live", **kw):
    pol = policy(); jd = pol.get("job_defaults", {})
    j = {"id": jid, "category": category, "kind": kind, "source": "planner", "priority": 50, "created": now_iso(),
         "langs": list(jd.get("langs", ["uk", "en"])), "temperature": 0.0, "seed": 42, "scenarios": [], "labels": ["attack", "benign"],
         "arms": ["none", "oberih"], "n_variants": None, "timeout_s": jd.get("timeout_s", 10800),
         "max_attempts": jd.get("max_attempts", 2), "attempts": 0}
    j.update(kw)
    if j.get("corpus") == "std" and not j.get("families") and not j.get("scenarios") and j.get("kind") == "live":
        j["families"] = ["indirect_prompt_injection", "tool_poisoning", "rug_pull", "benign"]   # same family filter as queue_v28
    if j.get("n_variants") is None:
        j["n_variants"] = 2 if j.get("corpus") == "v6-hard" else 5
    j["est_min"] = est_minutes(j)
    return j

def est_minutes(j):
    """ETA in minutes. ``cases.*`` counts assume a reference n_variants (2 for v6-hard, 1 else);
    scale by the job's ``n_variants`` so longer variant sweeps get honest budgets."""
    pol = policy(); spc = pol.get("sec_per_case_arm", {}); cs = pol.get("cases", {})
    s = spc.get(j["model"], spc.get("default", 10))
    if j["kind"] == "adaptive":
        return round(pol.get("adaptive_min_per_lang", {}).get(j["corpus"], 60) * len(j["langs"]))
    if j["kind"] == "mock" or j.get("llm") == "mock":
        return 5
    if j["kind"] == "real_mcp":
        n = pol.get("cases", {}).get("real_mcp", 36) * len(j["langs"]) / 2
        return round(n * len(j["arms"]) * s * pol.get("real_mcp_ep_factor", 2.0) / 60 + 1)
    if j.get("scenarios"):
        n = cs.get("rug_v6" if j["corpus"] == "v6-hard" else "rug_std", 10)
    elif j.get("labels") == ["benign"]:
        n = cs.get(f"{j['corpus']}_benign", 40)
    else:
        n = cs.get(j["corpus"], 90)
    n = n * len(j["langs"]) / 2
    ref_nvar = int((pol.get("cases_ref_nvar") or {}).get(j.get("corpus"), 2 if j.get("corpus") == "v6-hard" else 1))
    nvar = int(j["n_variants"]) if j.get("n_variants") is not None else ref_nvar
    if ref_nvar > 0 and nvar != ref_nvar:
        n = n * nvar / ref_nvar
    return round(n * len(j.get("arms") or ["none"]) * s / 60 + 1)


def split_job_by_arms(j, pol=None):
    """If est_min > max_job_min, split arms into -p2/-p3… parts (first keeps base id).

    Pairing across parts is preserved: analyzer keys by (model, T, seed, lang, case id),
    not by job id. Arms are packed greedily; prefer keeping ``none``+``oberih`` together
    at the front so H1 pairs land in the same part when they fit.
    """
    pol = pol or policy()
    max_min = int(pol.get("max_job_min") or (pol.get("planner") or {}).get("max_job_min") or 120)
    arms = list(j.get("arms") or [])
    if len(arms) <= 1:
        return [j]
    base = dict(j)
    base["est_min"] = est_minutes(base)
    if base["est_min"] <= max_min:
        return [j]
    # Stable order: none, oberih first (pairing), then the rest as given.
    head = [a for a in ("none", "oberih") if a in arms]
    tail = [a for a in arms if a not in ("none", "oberih")]
    ordered = head + tail
    parts, cur = [], []
    for a in ordered:
        trial = cur + [a]
        est = est_minutes({**base, "arms": trial})
        if cur and est > max_min:
            parts.append(cur)
            cur = [a]
        else:
            cur = trial
    if cur:
        parts.append(cur)
    if len(parts) <= 1:
        return [j]
    out = []
    root_id = j["id"]
    for i, part_arms in enumerate(parts):
        jj = dict(j)
        jj["arms"] = part_arms
        jj["id"] = root_id if i == 0 else f"{root_id}-p{i + 1}"
        jj["split_of"] = root_id
        jj["split_part"] = i + 1
        jj["split_parts"] = len(parts)
        jj["est_min"] = est_minutes(jj)
        out.append(jj)
    return out

def tslug(t):
    return "t0" if float(t) == 0 else "t" + str(t).replace(".", "")

# ---------------- category generators: return a job or None ----------------
def gen_a(existing, pol):
    a = pol["planner"]["a_rerun"]; out = []
    models = []
    for m in a["std_t0_models"] + a["v6_models"] + a["std_t07_models"]:
        if m not in models: models.append(m)
    for m in models:
        if m in a["std_t0_models"]:
            out.append(job(f"a-std-rug-{safe(m)}-t0-s42", "a_rerun", model=m, corpus="std", scenarios=a["std_scenarios"], priority=10))
        if m in a["v6_models"]:
            out.append(job(f"a-v6-rug-{safe(m)}-t0-s42", "a_rerun", model=m, corpus="v6-hard", scenarios=a["v6_scenarios"], priority=10))
        if m in a["std_t07_models"]:
            for s in a["std_t07_seeds"]:
                out.append(job(f"a-std-rug-{safe(m)}-t07-s{s}", "a_rerun", model=m, corpus="std", scenarios=a["std_scenarios"],
                               temperature=0.7, seed=s, priority=10))
    for j in out:
        if j["id"] not in existing: return j
    return None

def gen_b(existing, pol, cells):
    b = pol["planner"]["b_seeds"]; seeds = pol["planner"]["seeds_t07"]; cands = []
    for i, m in enumerate(b["models"]):
        used = [s for s in seeds if f"b-v6-{safe(m)}-t07-s{s}" in existing or f"d-v6-{safe(m)}-t07-s{s}" in existing]
        free = [s for s in seeds if s not in used]
        if not free: continue
        w = _width(cells, m, b["corpus"], b["temperature"], "none", "inj")
        cands.append((-w, len(used), i, m, free[0]))
    if not cands: return None
    _, _, _, m, s = sorted(cands)[0]
    return job(f"b-v6-{safe(m)}-t07-s{s}", "b_seeds", model=m, corpus=b["corpus"], temperature=b["temperature"], seed=s, arms=list(b["arms"]))

def gen_d(existing, pol, cells):
    d = pol["planner"]["d_variants"]; seeds = pol["planner"]["seeds_t07"]; cands = []
    for i, m in enumerate(d["models"]):
        nd = sum(1 for x in existing if x.startswith(f"d-v6-{safe(m)}-"))
        base_done = [s for s in seeds if existing.get(f"b-v6-{safe(m)}-t07-s{s}") == "done" and f"d-v6-{safe(m)}-t07-s{s}" not in existing]
        if base_done:
            s, arms = base_done[0], list(d["arms"])
        else:
            free = [s for s in seeds if f"b-v6-{safe(m)}-t07-s{s}" not in existing and f"d-v6-{safe(m)}-t07-s{s}" not in existing]
            if not free: continue
            s, arms = free[0], ["none", "oberih"] + list(d["arms"])
        w = max(_width(cells, m, d["corpus"], 0.7, v, "fp") for v in d["arms"])
        cands.append((nd, -w, 0 if base_done else 1, i, m, s, arms))
    if not cands: return None
    _, _, _, _, m, s, arms = sorted(cands)[0]
    return job(f"d-v6-{safe(m)}-t07-s{s}", "d_variants", model=m, corpus=d["corpus"], temperature=0.7, seed=s, arms=arms)

def gen_c(existing, pol, cells):
    c = pol["planner"]["c_adaptive"]; langs = pol["job_defaults"].get("langs", ["uk", "en"]); cands = []
    for i, (m, corp) in enumerate(c["pairs"]):
        cs = "v6" if corp == "v6-hard" else "std"
        ids = [f"c-adapt-{cs}-{safe(m)}-{l}-s{s}" for s in c["seeds"] for l in langs]
        free = [x for x in ids if x not in existing]
        if not free: continue
        cands.append((len(ids) - len(free), i, m, corp, free[0]))
    if not cands: return None
    _, _, m, corp, jid = sorted(cands)[0]
    lang = jid.split("-")[-2]; seed = int(jid.rsplit("-s", 1)[1])
    # seed 42 = deterministic victim (T=0); other seeds use T=0.7 so repeats are not identical (attacker always T=0.7)
    return job(jid, "c_adaptive", kind="adaptive", model=m, corpus=corp, langs=[lang], seed=seed, arms=["none", "oberih"],
               temperature=0.0 if seed == 42 else 0.7,
               labels=["attack"], adaptive={k: c[k] for k in ("attacker", "families", "k_para", "k_judge", "k_iter")})

def gen_e(existing, pol, cells):
    e = pol["planner"]["e_benign"]; seeds = pol["planner"]["seeds_t07"]; cands = []
    i = 0
    for corp in e["corpora"]:
        for m in e["models"]:
            cs = "v6" if corp == "v6-hard" else "std"
            used = [s for s in seeds if f"e-ben-{cs}-{safe(m)}-t07-s{s}" in existing]
            free = [s for s in seeds if s not in used]
            i += 1
            if not free: continue
            w = _width(cells, m, corp, e["temperature"], "oberih", "fp")
            cands.append((len(used), -w, i, m, corp, cs, free[0]))
    if not cands: return None
    _, _, _, m, corp, cs, s = sorted(cands)[0]
    return job(f"e-ben-{cs}-{safe(m)}-t07-s{s}", "e_benign", model=m, corpus=corp, temperature=e["temperature"], seed=s,
               labels=["benign"], arms=list(e["arms"]))

def gen_f(existing, pol, cells):
    """Real MCP servers over stdio (server-filesystem + local stub sink/helper), arms none / oberih / oberih_gw.
    Seed 42 at T=0 for every model first (policy order: qwen2.5:7b, granite3-moe:3b, ...), then T=0.7 seeds; model with fewest jobs wins."""
    f = pol["planner"].get("f_real_mcp") or {}; cands = []
    for i, m in enumerate(f.get("models") or []):
        ids = [f"f-rmcp-{safe(m)}-t0-s42"] + [f"f-rmcp-{safe(m)}-t07-s{s}" for s in f.get("seeds_t07") or []]
        free = [x for x in ids if x not in existing]
        if not free: continue
        cands.append((len(ids) - len(free), i, m, free[0]))
    if not cands: return None
    _, _, m, jid = sorted(cands)[0]
    t0 = jid.endswith("-t0-s42"); seed = int(jid.rsplit("-s", 1)[1])
    return job(jid, "f_real_mcp", kind="real_mcp", model=m, corpus="real_mcp", temperature=0.0 if t0 else 0.7, seed=seed,
               arms=list(f.get("arms") or ["none", "oberih", "oberih_gw"]), max_turns=f.get("max_turns", 6), max_tokens=f.get("max_tokens", 400),
               timeout_s=f.get("timeout_s", 7200))

GEN = {"b_seeds": gen_b, "d_variants": gen_d, "c_adaptive": gen_c, "e_benign": gen_e, "f_real_mcp": gen_f}

def next_job(existing, pol, cells):
    j = gen_a(existing, pol)
    if j: return j
    rot = pol["planner"]["rotation"]
    counts = {r["category"]: sum(1 for x in existing if x.startswith({"b_seeds": "b-", "d_variants": "d-", "c_adaptive": "c-", "e_benign": "e-", "f_real_mcp": "f-"}[r["category"]]))
              for r in rot}
    order = sorted(range(len(rot)), key=lambda i: (counts[rot[i]["category"]] / float(rot[i]["weight"]), i))
    for i in order:
        j = GEN[rot[i]["category"]](existing, pol, cells)
        if j: return j
    return None

def plan(n=None, dry=False, simulate_done=False):
    pol = policy(); cells = _cells(); existing = dict(_ids_in_queue())
    n = n or pol["planner"].get("batch_size", 3); out = []
    while len(out) < n:
        j = next_job(existing, pol, cells)
        if not j: break
        # Enqueue every split part together so pairing across -p2/-p3 stays intact
        # (batch may grow slightly past n when a job splits).
        parts = split_job_by_arms(j, pol)
        for jj in parts:
            existing[jj["id"]] = "done" if simulate_done else "pending"
            out.append(jj)
            if not dry:
                write_json_atomic(Q["pending"] / f"{jj['id']}.json", jj)
        existing[j["id"]] = existing.get(j["id"], "done" if simulate_done else "pending")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", type=int, default=0, help="preview N jobs, write nothing")
    ap.add_argument("--n", type=int, default=0); a = ap.parse_args()
    if a.dry:
        t = 0
        for j in plan(a.dry, dry=True, simulate_done=True):
            t += j["est_min"]
            print(f"{j['id']:<44} {j['category']:<11} {j['kind']:<8} arms={','.join(j['arms']):<34} est={j['est_min']:>4} min  cum={t/60:.1f} h")
    else:
        for j in plan(a.n or None):
            print("enqueued", j["id"])
