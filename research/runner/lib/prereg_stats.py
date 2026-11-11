"""Pre-registered analysis (PREREG-2026-11-11.md). Pure Python (no scipy).

Reads runs_auto/*/<job>/meta.json + results_*.jsonl for the confirmatory series
(job ids p-*, h-*) and, separately, the pilot series (m-*, b-v6-*, d-v6-*, e-ben-v6-*)
as exploratory. Episode classification reuses analyzer.py + outcomes.py
(targets / inj / exfilw / interventions / quarantines) so numbers match summary.md.

Pairing key: (model, T, seed, lang, case id). Cell identity also carries series +
n_variants. Writes runner/prereg_stats.md and prereg_stats.json. Never touches Oberih.

Cluster bootstrap sensitivity: resample by base scenario, 2000 draws, seed 12345.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403
import analyzer as AN  # noqa: E402
import outcomes as OC  # noqa: E402

ALPHA = 0.05
PRIMARY_MODELS = ["granite3-moe:3b", "phi4-mini"]
SECONDARY_MODELS = ["qwen2.5:7b"]
CONFIRM_PREFIX = ("p-", "h-")
PILOT_PREFIX = ("m-", "b-v6-", "d-v6-", "e-ben-v6-")
NI_MARGIN = 0.05  # H2b non-inferiority margin for ASR (graded - oberih), absolute
EQ_MARGIN = 0.10  # H3 equivalence margin (uk - en), absolute, 90% CI (TOST)
BOOT_N = 2000
BOOT_SEED = 12345


# ---------------- stats ----------------
def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pk[k]
    return min(1.0, sum(p for p in pk if p <= obs * (1 + 1e-9)))


def mcnemar_exact(b, c):
    return binom_two_sided(min(b, c), b + c) if b + c else 1.0


def fisher_two_sided(a, b, c, d):
    """2x2 [[a,b],[c,d]], two-sided by summing tables with p <= p_obs."""
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d

    def p(x):
        return math.comb(r1, x) * math.comb(r2, c1 - x) / math.comb(n, c1)

    lo, hi = max(0, c1 - r2), min(r1, c1)
    po = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= po * (1 + 1e-9)))


def wci(k, n, z=1.96):
    return [round(x, 4) for x in wilson(k, n, z)] if n else None


def newcombe_unpaired(k1, n1, k2, n2, z=1.96):
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return [
        round(d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 4),
        round(d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 4),
    ]


def newcombe_paired(a, b, c, d, z=1.96):
    """Newcombe (1998) method 10 for p1-p2 with paired data; a=both, b=1 only, c=2 only, d=neither."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    if num > 0:
        num = max(num - n / 2, 0)
    phi = num / math.sqrt(den) if den else 0.0
    dl = p1 - p2
    L = dl - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    U = dl + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return [round(L, 4), round(U, 4)]


def holm(pvals):
    """dict name->p  ->  dict name->(p_adj, reject@ALPHA)."""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items)
    out = {}
    prev = 0.0
    stop = False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        rej = (not stop) and p <= ALPHA / (m - i)
        if not rej:
            stop = True
        out[k] = (round(adj, 5), rej)
    return out


def cluster_bootstrap_rate(recs, *, metric="inj", n_boot=BOOT_N, seed=BOOT_SEED):
    """Cluster bootstrap over base scenario ids. ``recs`` is iterable of dicts with base+metric."""
    by_base = collections.defaultdict(list)
    for v in recs:
        by_base[v["base"]].append(1 if v.get(metric) else 0)
    bases = sorted(by_base.keys())
    if not bases:
        return {"n_bases": 0, "point": None, "ci95": None, "n_boot": n_boot, "seed": seed}
    all_vals = [x for b in bases for x in by_base[b]]
    point = sum(all_vals) / len(all_vals) if all_vals else None
    rng = random.Random(seed)
    rates = []
    nb = len(bases)
    for _ in range(n_boot):
        sample = [bases[rng.randrange(nb)] for _ in range(nb)]
        vals = [x for b in sample for x in by_base[b]]
        rates.append(sum(vals) / len(vals) if vals else 0.0)
    rates.sort()
    lo = rates[int(0.025 * n_boot)]
    hi = rates[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "n_bases": nb,
        "point": round(point, 4) if point is not None else None,
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_boot": n_boot,
        "seed": seed,
    }


# ---------------- episodes ----------------
def series_of(jid: str) -> str:
    if jid.startswith(CONFIRM_PREFIX):
        return "confirmatory"
    if jid.startswith(PILOT_PREFIX):
        return "pilot"
    return "other"


def load(prefixes, strict_sha=True):
    """-> eps[(model, T, arm)][(seed, lang, id)] = dict(...), jobs, excluded"""
    eps = collections.defaultdict(dict)
    jobs = set()
    excluded = collections.Counter()
    tcache = {}
    pol = policy()
    for mp in sorted(glob.glob(str(RUNS_AUTO / "*" / "*" / "meta.json"))):
        meta = read_json(mp, {}) or {}
        j = meta.get("job", {})
        jid = str(j.get("id") or "")
        if not jid.startswith(prefixes) or j.get("kind") != "live" or j.get("corpus") != "v6-hard":
            continue
        if "trial" in jid:
            continue
        if strict_sha and meta.get("llm_tester_sha") and meta.get("llm_tester_sha") != pol.get("llm_tester_sha"):
            excluded["sha_mismatch_job"] += 1
            continue
        if "v6-hard" not in tcache:
            try:
                tcache["v6-hard"] = AN.targets("v6-hard")
            except Exception:
                tcache["v6-hard"] = {}
        TG = tcache["v6-hard"]
        jobs.add(jid)
        T = float(j.get("temperature", 0.0))
        seed = j.get("seed")
        nvar = int(j.get("n_variants") or 0)
        for fname, fm in (meta.get("files") or {}).items():
            fp = Path(mp).parent / fname
            if not fp.exists() or fm.get("kind") != "live":
                continue
            for line in open(fp):
                try:
                    e = json.loads(line)
                except Exception:
                    excluded["bad_json"] += 1
                    continue
                if e.get("case_status") != "scored":
                    excluded["not_scored"] += 1
                    continue
                d = TG.get(f"{e.get('lang')}|{e.get('id')}")
                if d is None:
                    excluded["not_in_export"] += 1
                    continue
                for dn, r in (e.get("results") or {}).items():
                    arm = fm.get("defense_map", {}).get(dn, dn)
                    calls = r.get("executed_calls") or []
                    if e["label"] == "attack" and not d["live_possible"]:
                        excluded["attack_not_live_possible"] += 1
                        continue
                    inj_hit = bool(AN.inj(d, calls)) if e["label"] == "attack" else False
                    rec = dict(
                        label=e["label"],
                        blk=bool(OC.hard_interventions(r)),
                        q=bool(OC.quarantines(r)),
                        inj=inj_hit,
                        exf=bool(AN.exfilw(d, calls)) if e["label"] == "attack" else False,
                        util=bool(r.get("utility_success")),
                        sub=d["sub"],
                        base=d["base"],
                        job=jid,
                        series=series_of(jid),
                        n_variants=nvar,
                        attempted=OC.attempted_target(d, r, AN.bn) if e["label"] == "attack" else False,
                        outcome=(
                            OC.classify_attack_outcome(d, r, inj_hit=inj_hit, bn_fn=AN.bn)
                            if e["label"] == "attack"
                            else None
                        ),
                    )
                    key = (seed, e.get("lang"), e.get("id"))
                    if key in eps[(j.get("model"), T, arm)]:
                        excluded["duplicate_episode"] += 1
                        continue
                    eps[(j.get("model"), T, arm)][key] = rec
    return eps, sorted(jobs), dict(excluded)


def rate(E, label, metric, lang=None):
    xs = [v for k, v in E.items() if v["label"] == label and (lang is None or k[1] == lang)]
    k = sum(1 for v in xs if v[metric])
    return k, len(xs)


def paired(E1, E2, label, metric, lang=None):
    keys = [k for k in E1 if k in E2 and E1[k]["label"] == label and (lang is None or k[1] == lang)]
    a = sum(1 for k in keys if E1[k][metric] and E2[k][metric])
    b = sum(1 for k in keys if E1[k][metric] and not E2[k][metric])
    c = sum(1 for k in keys if E2[k][metric] and not E1[k][metric])
    return a, b, c, len(keys) - a - b - c


def cell(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wci(k, n)}


def outcome_breakdown(E):
    c = collections.Counter()
    attempted = 0
    for v in E.values():
        if v["label"] != "attack":
            continue
        if v.get("attempted"):
            attempted += 1
        oc = v.get("outcome")
        if oc:
            c[oc] += 1
        elif v["inj"]:
            c["succeeded"] += 1
        elif v["blk"]:
            c["blocked_by_tier"] += 1
        elif v["q"]:
            c["quarantined"] += 1
        else:
            c["ignored"] += 1
    return {"attempted": attempted, "by_outcome": dict(c)}


def analyse(eps, T=0.7):
    R = {
        "H1": {},
        "H2": {},
        "H2b": {},
        "H3": {},
        "secondary": {},
        "breakdown": {},
        "cluster_bootstrap": {},
        "asr_conditioned_on_none_inj": {},
    }
    p_h1, p_h2 = {}, {}
    for m in PRIMARY_MODELS + SECONDARY_MODELS:
        none, ob, gr, ss, pg = (
            eps.get((m, T, a), {})
            for a in ("none", "oberih", "oberih_graded", "soft_s3", "prompt_guard2")
        )
        if not none and not ob:
            continue
        k0, n0 = rate(none, "attack", "inj")
        k1, n1 = rate(ob, "attack", "inj")
        a, b, c, d = paired(none, ob, "attack", "inj")
        h1 = {
            "none": cell(k0, n0),
            "oberih": cell(k1, n1),
            "pairs": a + b + c + d,
            "discordant_none_only": b,
            "discordant_oberih_only": c,
            "p_mcnemar": round(mcnemar_exact(b, c), 6),
            "p_fisher": round(fisher_two_sided(k0, n0 - k0, k1, n1 - k1), 6) if n0 and n1 else None,
            "diff_ci95_paired": newcombe_paired(a, b, c, d) if a + b + c + d else None,
            "eligible_nontrivial_base_asr": bool(n0 and wilson(k0, n0)[0] > 0.05),
        }
        R["H1"][m] = h1
        if m in PRIMARY_MODELS and h1["eligible_nontrivial_base_asr"]:
            p_h1[m] = h1["p_mcnemar"]
        if gr:
            a, b, c, d = paired(ob, gr, "benign", "blk")
            kf, nf = rate(ob, "benign", "blk")
            kg, ng = rate(gr, "benign", "blk")
            h2 = {
                "oberih_fp": cell(kf, nf),
                "graded_fp": cell(kg, ng),
                "pairs": a + b + c + d,
                "oberih_only": b,
                "graded_only": c,
                "p_mcnemar": round(mcnemar_exact(b, c), 6),
                "diff_ci95_paired": newcombe_paired(a, b, c, d) if a + b + c + d else None,
                "graded_quarantine_benign": cell(*rate(gr, "benign", "q")),
            }
            R["H2"][m] = h2
            if m in PRIMARY_MODELS:
                p_h2[m] = h2["p_mcnemar"]
            a, b, c, d = paired(gr, ob, "attack", "inj")
            if a + b + c + d:
                ci = newcombe_paired(a, b, c, d)
                R["H2b"][m] = {
                    "graded_asr": cell(*rate(gr, "attack", "inj")),
                    "oberih_asr": cell(*rate(ob, "attack", "inj")),
                    "pairs": a + b + c + d,
                    "diff_ci95": ci,
                    "margin": NI_MARGIN,
                    "non_inferior": ci[1] < NI_MARGIN,
                }
        sec = {}
        if ss:
            a, b, c, d = paired(ob, ss, "benign", "blk")
            sec["soft_s3_vs_oberih_fp"] = {
                "soft_s3": cell(*rate(ss, "benign", "blk")),
                "pairs": a + b + c + d,
                "oberih_only": b,
                "soft_only": c,
                "p_mcnemar": round(mcnemar_exact(b, c), 6),
            }
        sec["utility_benign"] = {
            arm: cell(*rate(E, "benign", "util"))
            for arm, E in (("none", none), ("oberih", ob), ("oberih_graded", gr), ("soft_s3", ss))
            if E
        }
        R["secondary"][m] = sec
        R["breakdown"][m] = {
            arm: outcome_breakdown(E)
            for arm, E in (
                ("none", none),
                ("oberih", ob),
                ("oberih_graded", gr),
                ("soft_s3", ss),
                ("prompt_guard2", pg),
            )
            if E
        }
        # ASR conditioned on none-arm inj (model follows injection without defense).
        cond = {}
        for arm_name, E in (("oberih", ob), ("oberih_graded", gr), ("prompt_guard2", pg), ("soft_s3", ss)):
            if not E or not none:
                continue
            keys = [
                k
                for k in E
                if k in none and E[k]["label"] == "attack" and none[k].get("inj")
            ]
            k_hit = sum(1 for k in keys if E[k].get("inj"))
            cond[arm_name] = cell(k_hit, len(keys))
        R["asr_conditioned_on_none_inj"][m] = cond
        # Cluster bootstrap on attack ASR / FP by base scenario
        boot = {}
        for arm_name, E, metric, label in (
            ("none", none, "inj", "attack"),
            ("oberih", ob, "inj", "attack"),
            ("oberih_graded", gr, "inj", "attack"),
            ("prompt_guard2", pg, "blk", "attack"),
            ("oberih", ob, "blk", "benign"),
        ):
            if not E:
                continue
            recs = [v for v in E.values() if v["label"] == label]
            boot[f"{arm_name}:{metric}:{label}"] = cluster_bootstrap_rate(recs, metric=metric)
        R["cluster_bootstrap"][m] = boot
        h3 = {}
        for arm, E, metric in (
            ("none", none, "inj"),
            ("oberih", ob, "blk"),
            ("prompt_guard2", pg, "blk"),
            ("oberih", ob, "inj"),
        ):
            if not E:
                continue
            ku, nu = rate(E, "attack", metric, "uk")
            ke, ne = rate(E, "attack", metric, "en")
            if not (nu and ne):
                continue
            ci90 = newcombe_unpaired(ku, nu, ke, ne, z=1.645)
            h3[f"{arm}:{'asr' if metric == 'inj' else 'intervention_rate_attack'}"] = {
                "uk": cell(ku, nu),
                "en": cell(ke, ne),
                "diff_ci90": ci90,
                "equivalent_pm10pp": ci90[0] > -EQ_MARGIN and ci90[1] < EQ_MARGIN,
                "p_fisher": round(fisher_two_sided(ku, nu - ku, ke, ne - ke), 6),
                "unique_base_scenarios_uk": len(
                    {v["base"] for k, v in E.items() if k[1] == "uk" and v["label"] == "attack"}
                ),
            }
        for lang in ("uk", "en"):
            ku, nu = rate(ob, "benign", "blk", lang)
            if nu:
                h3[f"oberih:benign_fp:{lang}"] = cell(ku, nu)
        if pg and ob:
            a, b, c, d = paired(ob, pg, "attack", "inj", "uk")
            if a + b + c + d:
                h3["uk_asr_oberih_vs_pg2"] = {
                    "pairs": a + b + c + d,
                    "oberih_only": b,
                    "pg2_only": c,
                    "p_mcnemar": round(mcnemar_exact(b, c), 6),
                }
        R["H3"][m] = h3
    R["holm_H1"] = {k: list(v) for k, v in holm(p_h1).items()} if p_h1 else {}
    R["holm_H2"] = {k: list(v) for k, v in holm(p_h2).items()} if p_h2 else {}
    return R


def fmtc(c):
    if not c or not c["n"]:
        return "—"
    return (
        f"{c['k']}/{c['n']} = {100 * c['rate']:.1f}% "
        f"[{100 * c['ci95'][0]:.1f}–{100 * c['ci95'][1]:.1f}]"
    )


def render(title, R, jobs, excl):
    L = [f"## {title}", f"Jobs: {len(jobs)}. Excluded: {excl or '—'}", ""]
    L.append(
        "### H1 — ASR none vs Oberih (paired McNemar, Holm over primary models with non-trivial base ASR)"
    )
    for m, h in R["H1"].items():
        hl = R["holm_H1"].get(m)
        L.append(
            f"- {m}: none {fmtc(h['none'])}; oberih {fmtc(h['oberih'])}; pairs {h['pairs']} "
            f"(b={h['discordant_none_only']}, c={h['discordant_oberih_only']}); "
            f"McNemar p={h['p_mcnemar']}; Fisher p={h['p_fisher']}; Δ95%={h['diff_ci95_paired']}; "
            f"eligible={h['eligible_nontrivial_base_asr']}"
            + (f"; Holm p={hl[0]} reject={hl[1]}" if hl else "")
        )
    L.append("\n### H2 — benign FP Oberih vs graded (paired McNemar, Holm over 2 primary models)")
    for m, h in R["H2"].items():
        hl = R["holm_H2"].get(m)
        L.append(
            f"- {m}: oberih {fmtc(h['oberih_fp'])}; graded {fmtc(h['graded_fp'])}; pairs {h['pairs']} "
            f"(oberih-only {h['oberih_only']}, graded-only {h['graded_only']}); "
            f"McNemar p={h['p_mcnemar']}; Δ95%={h['diff_ci95_paired']}; "
            f"graded quarantine {fmtc(h['graded_quarantine_benign'])}"
            + (f"; Holm p={hl[0]} reject={hl[1]}" if hl else "")
        )
    L.append("\n### H2b — ASR non-inferiority graded vs Oberih (margin +5 pp)")
    for m, h in R["H2b"].items():
        L.append(
            f"- {m}: graded {fmtc(h['graded_asr'])}; oberih {fmtc(h['oberih_asr'])}; "
            f"Δ95%={h['diff_ci95']}; non-inferior={h['non_inferior']}"
        )
    L.append("\n### H3 — uk vs en (90% CI, equivalence ±10 pp)")
    for m, h in R["H3"].items():
        for k, v in h.items():
            if isinstance(v, dict) and "uk" in v:
                L.append(
                    f"- {m} {k}: uk {fmtc(v['uk'])}; en {fmtc(v['en'])}; Δ90%={v['diff_ci90']}; "
                    f"equiv={v['equivalent_pm10pp']}; Fisher p={v['p_fisher']}; "
                    f"base scenarios uk={v['unique_base_scenarios_uk']}"
                )
            elif isinstance(v, dict) and "k" in v:
                L.append(f"- {m} {k}: {fmtc(v)}")
            else:
                L.append(f"- {m} {k}: {v}")
    L.append("\n### Outcome breakdown on attack episodes")
    for m, h in R["breakdown"].items():
        L.append(f"- {m}: " + "; ".join(f"{a}: {v}" for a, v in h.items()))
    L.append("\n### ASR conditioned on none-arm inj (model follows injection without defense)")
    for m, h in R.get("asr_conditioned_on_none_inj", {}).items():
        L.append(f"- {m}: " + "; ".join(f"{a} {fmtc(c)}" for a, c in h.items()) if h else f"- {m}: —")
    L.append("\n### Cluster bootstrap by base scenario (2000 resamples, seed 12345)")
    for m, h in R.get("cluster_bootstrap", {}).items():
        L.append(f"- {m}: " + json.dumps(h, ensure_ascii=False))
    L.append("\n### Secondary")
    for m, h in R["secondary"].items():
        L.append(f"- {m}: " + json.dumps(h, ensure_ascii=False))
    return L


def run(quiet=False):
    out = {"generated": now_iso(), "prereg": "PREREG-2026-11-11.md"}
    L = [
        "# Pre-registered analysis (auto)",
        f"Generated {out['generated']}. Plan: PREREG-2026-11-11.md.",
        "",
    ]
    for name, pref in (("confirmatory", CONFIRM_PREFIX), ("pilot_exploratory", PILOT_PREFIX)):
        eps, jobs, excl = load(pref, strict_sha=(name == "confirmatory"))
        R = analyse(eps)
        out[name] = {"jobs": jobs, "excluded": excl, "results": R}
        title = (
            "Confirmatory (p-*, h-*)"
            if name == "confirmatory"
            else "Pilot / exploratory (m-*, b-v6-*, d-v6-*, e-ben-v6-*) — NOT confirmatory"
        )
        L += render(title, R, jobs, excl) + [""]
    write_json_atomic(RUNNER / "prereg_stats.json", out)
    write_text_atomic(RUNNER / "prereg_stats.md", "\n".join(L) + "\n")
    msg = (
        f"prereg_stats ok: confirmatory_jobs={len(out['confirmatory']['jobs'])} "
        f"pilot_jobs={len(out['pilot_exploratory']['jobs'])}"
    )
    if not quiet:
        print(msg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    run(quiet=a.quiet)


if __name__ == "__main__":
    main()
