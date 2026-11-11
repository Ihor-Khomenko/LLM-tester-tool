"""Oberih 24/7 runner supervisor (launchd label default: org.oberih.runner). File queue: queue/pending -> running -> done|failed.
One job at a time, one model in memory. Guards: legacy queue PIDs, PAUSE file, Ollama up, free disk, swap, battery.
Resumable drivers (append + skip done rows by run_tag) make preemption safe: a preempted job goes back to pending
without spending an attempt. Never deletes Ollama models, never calls paid APIs, never pushes to git."""
import os, sys, json, time, signal, subprocess, shutil, gzip, fcntl, argparse, traceback, socket
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import planner

LOG = RUNNER / "logs" / "supervisor.log"
STATE = {"last_finished": None, "battery_latched": False, "started": now_iso(), "current": None, "last_plan": None}
HB = 15  # seconds between heartbeats while a job runs

def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"{now_iso()} {msg}\n")

def rotate(p, max_bytes=2_000_000, keep=5):
    p = Path(p)
    try:
        if p.exists() and p.stat().st_size > max_bytes:
            for i in range(keep - 1, 0, -1):
                a, b = p.with_name(f"{p.name}.{i}"), p.with_name(f"{p.name}.{i+1}")
                if a.exists(): os.replace(a, b)
            shutil.copyfile(p, p.with_name(p.name + ".1")); open(p, "w").close()
    except Exception as e:
        log(f"rotate {p}: {e!r}")

def guards(pol, check_legacy=True):
    g = pol.get("guards", {}); b = []
    fg, sw = free_gb(), swap_used_mb(); src, pct = battery()
    if check_legacy:
        for x in pol.get("legacy_wait") or []:
            if pid_alive(x["pid"], x.get("cmd")): b.append(f"waiting_legacy:{x['pid']}({x.get('cmd')})")
    if (RUNNER / "PAUSE").exists(): b.append("paused_by_user")
    if fg < g.get("min_free_gb", 22): b.append(f"disk_free_{fg:.1f}GB<{g.get('min_free_gb', 22)}")
    if sw > g.get("max_swap_mb", 12000): b.append(f"swap_{sw:.0f}MB>{g.get('max_swap_mb', 12000)}")
    if src == "Battery" and pct is not None and pct < g.get("battery_min_pct", 30): STATE["battery_latched"] = True
    if src == "AC" or g.get("battery_min_pct", 30) <= 0: STATE["battery_latched"] = False
    if STATE["battery_latched"]: b.append(f"battery_{pct}%<{g.get('battery_min_pct', 30)}_wait_for_AC")
    if g.get("require_ollama", True) and not ollama_up(): b.append("ollama_down")
    return b, {"free_gb": round(fg, 1), "swap_mb": round(sw), "power": src, "battery_pct": pct}

def heartbeat(pol, state, blockers, env, extra=None):
    qs = {k: len(list(Q[k].glob("*.json"))) for k in Q}
    st = {"ts": now_iso(), "host": socket.gethostname(), "pid": os.getpid(), "state": state, "blockers": blockers,
          "guard_state": env, "guard_thresholds": pol.get("guards", {}), "queue": qs, "current": STATE["current"],
          "last_finished": STATE["last_finished"], "last_plan": STATE["last_plan"], "supervisor_started": STATE["started"],
          "ollama_loaded": ollama_loaded(), "noplan": (RUNNER / "NOPLAN").exists(), "llm_tester_sha": pol.get("llm_tester_sha")}
    if extra: st.update(extra)
    write_json_atomic(RUNNER / "status.json", st)

def child_env(pol):
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    lt = llm_tester_root(pol)
    env.update({"RESEARCH_ROOT": str(ROOT), "OBERIH_TESTS_ROOT": str(ROOT), "PYTHONPATH": str(lt),
                "LLM_TESTER_ROOT": str(lt), "LLM_TESTER_SHA": pol.get("llm_tester_sha", ""), "PYTHONUNBUFFERED": "1",
                "PATH": default_path_env(), "OLLAMA_HOST": os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")})
    return env

def py(pol):
    return venv_python(pol) or sys.executable

def steps_for(job, pol, outdir):
    """List of (step_name, out_file, cmd, defense_map, kind)."""
    PY = py(pol); L = Path(__file__).resolve().parent; jd = pol.get("job_defaults", {}); S = []
    langs = ",".join(job.get("langs") or ["uk", "en"])
    if job["kind"] == "mock":
        out = outdir / "results_mock.jsonl"
        S.append(("mock", out, [PY, str(L / "mock_driver.py"), "--out", str(out), "--run-tag", job["id"]] + (["--fail"] if job.get("mock_fail") else []),
                  {"none": "none", "sentinel": "oberih"}, "mock"))
        return S
    if job["kind"] == "real_mcp":
        out = outdir / "results_real_mcp.jsonl"; arms = list(job.get("arms") or ["none", "oberih", "oberih_gw"])
        cmd = [PY, str(L / "real_mcp_driver.py"), "--model", job["model"], "--llm", job.get("llm", "ollama"), "--langs", langs,
               "--arms", ",".join(arms), "--labels", ",".join(job.get("labels") or ["attack", "benign"]), "--ids", ",".join(job.get("scenarios") or []),
               "--temperature", str(job.get("temperature", 0.0)), "--seed", str(job.get("seed", 42)),
               "--max-turns", str(job.get("max_turns", 6)), "--max-tokens", str(job.get("max_tokens", 400)), "--run-tag", job["id"], "--out", str(out),
               "--protocol", job.get("protocol", "react")]
        S.append(("real_mcp", out, cmd, {a: a for a in arms}, "real_mcp"))
        return S
    if job["kind"] == "adaptive":
        a = job.get("adaptive") or {}; out = outdir / "results_adaptive.jsonl"
        cmd = [PY, str(L / "run_mcp_adaptive_ollama.py"), "--model", job["model"], "--attacker-model", a.get("attacker", "qwen2.5:7b"),
               "--corpus", "v6" if job["corpus"] == "v6-hard" else "std", "--langs", langs, "--defenses", "none,sentinel",
               "--families", a.get("families", "a_para,a_judge,a_iter"), "--k-para", str(a.get("k_para", 4)), "--k-judge", str(a.get("k_judge", 2)),
               "--k-iter", str(a.get("k_iter", 3)), "--temperature", str(job.get("temperature", 0.0)), "--seed", str(job.get("seed", 42)),
               "--max-turns", str(jd.get("max_turns", 4)), "--max-tokens", str(jd.get("max_tokens", 256)), "--run-tag", job["id"], "--out", str(out)]
        S.append(("adaptive", out, cmd, {"none": "none", "sentinel": "oberih"}, "adaptive"))
        return S
    # External baseline arms (not Oberih policy variants): pass defense name through.
    EXTERNAL = {"prompt_guard2", "prompt_guard", "pg2"}
    # Graded Oberih: dedicated defense id so response_mode=graded is set in _build_defense.
    GRADED = {"oberih_graded", "sentinel_graded", "graded"}
    base = [a for a in job["arms"] if a in ("none", "oberih")]
    groups = []
    if base: groups.append(("base", ",".join("none" if a == "none" else "sentinel" for a in base), {"none": "none", "sentinel": "oberih"}, {}))
    for a in job["arms"]:
        if a in ("none", "oberih"): continue
        al = a.lower().replace("-", "_")
        if al in EXTERNAL or a in EXTERNAL:
            dname = "prompt_guard2"
            groups.append((dname, dname, {dname: "prompt_guard2"}, {}))
            continue
        if al in GRADED or a in GRADED:
            groups.append(("oberih_graded", "oberih_graded", {"oberih_graded": "oberih_graded"}, {}))
            continue
        v = (pol.get("variants") or {}).get(a)
        if v is None: raise ValueError(f"unknown arm {a}")
        groups.append((a, "sentinel", {"sentinel": a}, v))
    for name, defs, dmap, v in groups:
        out = outdir / f"results_{name}.jsonl"
        cmd = [PY, str(L / "live_driver.py"), "--corpus", job["corpus"], "--nvar", str(job.get("n_variants") or 0),
               "--ids", ",".join(job.get("scenarios") or []), "--model", job["model"], "--langs", langs,
               "--labels", ",".join(job.get("labels") or ["attack", "benign"]), "--defenses", defs,
               "--temperature", str(job.get("temperature", 0.0)), "--seed", str(job.get("seed", 42)),
               "--max-turns", str(jd.get("max_turns", 4)), "--max-tokens", str(jd.get("max_tokens", 256)),
               "--run-tag", f"{job['id']}:{name}", "--out", str(out)]
        if job.get("families"): cmd += ["--families", ",".join(job["families"])]
        if v.get("soft_s3_allow"): cmd.append("--soft-s3-allow")
        if v.get("policy_json"): cmd += ["--policy-json", json.dumps(v["policy_json"], separators=(",", ":"))]
        S.append((name, out, cmd, dmap, "live"))
    return S

def nlines(p):
    try:
        with open(p, "rb") as f: return sum(1 for _ in f)
    except Exception: return 0

def corpus_sha(pol, corpus):
    d = ROOT / (pol.get("v6_corpus_dir") if corpus == "v6-hard" else pol.get("std_corpus_dir", "mcp-security-research"))
    for f in ("SHA.txt",):
        if (d / f).exists(): return (d / f).read_text().strip()[:64]
    v = ROOT / "VERSIONS.txt"
    if v.exists():
        for l in v.read_text().splitlines():
            if l.startswith("mcp-security-research ") and corpus != "v6-hard": return l.split()[1]
    return None

def move(job, state):
    src = Path(job.pop("_path")); dst = Q[state] / src.name
    write_json_atomic(dst, job)
    if src.exists() and src != dst: src.unlink()
    job["_path"] = str(dst); return dst

def finish(job, state, meta, outdir, pol, note):
    job["finished"] = now_iso(); job["last_status"] = state if state != "pending" else "retry"
    meta["status"] = {"done": "done", "failed": "failed", "pending": "retry_pending"}[state]; meta["finished"] = job["finished"]; meta["note"] = note
    write_json_atomic(outdir / "meta.json", meta)
    move(job, state)
    STATE["last_finished"] = {"id": job["id"], "status": meta["status"], "at": job["finished"], "note": note}
    log(f"FINISH {job['id']} -> {state} ({note})")
    if state in ("done", "failed"):
        lg = outdir / "run.log"
        if lg.exists() and lg.stat().st_size > 0:
            with open(lg, "rb") as fi, gzip.open(str(lg) + ".gz", "ab") as fo: shutil.copyfileobj(fi, fo)
            lg.unlink()
        try:
            r = subprocess.run([py(pol), str(Path(__file__).resolve().parent / "analyzer.py")], env=child_env(pol), capture_output=True, text=True, timeout=900)
            log(f"ANALYZER rc={r.returncode} {r.stdout.strip()[-200:]} {r.stderr.strip()[-400:]}")
        except Exception as e:
            log(f"ANALYZER failed {e!r}")

def stop_child(p):
    try:
        p.send_signal(signal.SIGTERM); p.wait(timeout=30)
    except Exception:
        try: p.kill(); p.wait(timeout=10)
        except Exception: pass

def run_job(job, pol, check_legacy=True):
    g = pol.get("guards", {})
    job = dict(job); job["attempts"] = int(job.get("attempts", 0)) + 1
    job.setdefault("first_started", now_iso()); job["started"] = now_iso()
    if not job.get("outdir"):
        job["outdir"] = str(RUNS_AUTO / job["first_started"][:10] / safe(job["id"]))
    move(job, "running")
    outdir = Path(job["outdir"]); outdir.mkdir(parents=True, exist_ok=True)
    meta = read_json(outdir / "meta.json", {}) or {}
    meta.update({"job": {k: v for k, v in job.items() if not k.startswith("_")}, "status": "running", "attempt": job["attempts"],
                 "llm_tester_sha": pol.get("llm_tester_sha"), "corpus_sha": corpus_sha(pol, job.get("corpus")),
                 "host": socket.gethostname(), "runner_policy_version": pol.get("version")})
    meta.setdefault("attempts_log", []).append({"attempt": job["attempts"], "started": job["started"]})
    log(f"START {job['id']} attempt={job['attempts']} kind={job['kind']} model={job.get('model')} out={outdir}")
    if job["kind"] != "mock" and job.get("llm") != "mock":
        if g.get("unload_between_jobs", True): ollama_unload_all()
        if not ollama_has(job["model"]):
            if job.get("allow_pull") and free_gb() >= g.get("min_free_gb", 22) + 10:
                log(f"PULL {job['model']}")
                subprocess.run([ollama_bin(), "pull", job["model"]], capture_output=True, timeout=3600, env=child_env(pol))
                if free_gb() < g.get("min_free_gb", 22):
                    log(f"WARN after pull free disk {free_gb():.1f}GB < min; guard will pause (models are never deleted)")
            if not ollama_has(job["model"]):
                job["last_error"] = "model not installed (planner never pulls; set allow_pull)"; job["attempts"] = job.get("max_attempts", 2)
                return finish(job, "failed", meta, outdir, pol, job["last_error"])
        if "tools" not in ollama_caps(job["model"]):
            job["last_error"] = "model has no tools capability"; job["attempts"] = job.get("max_attempts", 2)
            return finish(job, "failed", meta, outdir, pol, job["last_error"])
        meta["model_digest"], meta["model_size"] = ollama_digest(job["model"])
        if job.get("adaptive"): meta["attacker_digest"] = ollama_digest(job["adaptive"].get("attacker", "qwen2.5:7b"))[0]
    try:
        S = steps_for(job, pol, outdir)
    except Exception as e:
        job["last_error"] = f"bad job: {e!r}"; job["attempts"] = job.get("max_attempts", 2)
        return finish(job, "failed", meta, outdir, pol, job["last_error"])
    meta["files"] = {out.name: {"step": name, "kind": kind, "defense_map": dmap, "cmd": [c if not c.startswith(str(ROOT)) else c.replace(str(ROOT), "$ROOT") for c in cmd]}
                     for name, out, cmd, dmap, kind in S}
    meta.setdefault("timings", {})
    write_json_atomic(outdir / "meta.json", meta)
    # Each retry attempt gets a full fresh timeout budget (do not subtract prior elapsed_s).
    deadline = time.time() + int(job.get("timeout_s", 10800))
    env = child_env(pol); t_job = time.time()
    for name, out, cmd, dmap, kind in S:
        if (meta["timings"].get(name) or {}).get("rc") == 0:
            continue  # finished in an earlier attempt
        t0 = time.time(); lines0 = nlines(out)
        with open(outdir / "run.log", "a") as lf:
            lf.write(f"\n### {now_iso()} step={name} attempt={job['attempts']}\n"); lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT), start_new_session=True)
            STATE["current"] = {"id": job["id"], "step": name, "pid": p.pid, "started": job["started"], "est_min": job.get("est_min"),
                                "attempt": job["attempts"], "outdir": str(outdir), "category": job.get("category")}
            reason = None
            while True:
                try:
                    rc = p.wait(timeout=HB); break
                except subprocess.TimeoutExpired:
                    pass
                b, genv = guards(pol, check_legacy=False)
                b = [x for x in b if x not in ("paused_by_user",)]  # PAUSE lets the current job finish
                STATE["current"].update({"lines": nlines(out), "elapsed_min": round((time.time() - t_job) / 60, 1)})
                heartbeat(pol, "running", b, genv)
                if (RUNNER / "STOPNOW").exists(): reason = "stop_now"
                elif b and g.get("preempt_running_job", True) and any(x.split("_")[0] in ("disk", "swap", "battery") for x in b): reason = "guard:" + ";".join(b)
                elif time.time() > deadline: reason = "timeout"
                if reason:
                    stop_child(p); rc = p.returncode; break
        dt = time.time() - t0; job["elapsed_s"] = int(job.get("elapsed_s", 0) + dt)
        meta["timings"][name] = {"rc": rc, "seconds": round(dt), "lines_added": nlines(out) - lines0, "lines": nlines(out), "ended": now_iso(), "reason": reason}
        write_json_atomic(outdir / "meta.json", meta)
        STATE["current"] = None
        if job["kind"] != "mock" and g.get("unload_between_jobs", True): ollama_unload_all()
        if reason and reason != "timeout":
            (RUNNER / "STOPNOW").unlink(missing_ok=True)
            job["attempts"] -= 1; job["preempted"] = int(job.get("preempted", 0)) + 1
            return finish(job, "pending", meta, outdir, pol, f"preempted ({reason}); will resume")
        if kind == "adaptive" and rc == 5:
            job["attempts"] -= 1
            return finish(job, "pending", meta, outdir, pol, "adaptive driver stopped on disk<20GB; will resume")
        if rc != 0 or reason == "timeout":
            job["last_error"] = f"step {name} rc={rc} reason={reason}"
            st = "pending" if job["attempts"] < int(job.get("max_attempts", 2)) else "failed"
            return finish(job, st, meta, outdir, pol, job["last_error"] + ("; retry once" if st == "pending" else ""))
        if nlines(out) == 0:
            job["last_error"] = f"step {name} wrote no rows (scenario filter / export?)"; job["attempts"] = job.get("max_attempts", 2)
            return finish(job, "failed", meta, outdir, pol, job["last_error"])
    meta["elapsed_s"] = job["elapsed_s"]
    return finish(job, "done", meta, outdir, pol, "ok")

def recover_running():
    for j in list_jobs("running"):
        j["attempts"] = int(j.get("attempts", 1))
        log(f"RECOVER {j['id']} from running/ (supervisor restart) attempts={j['attempts']}")
        st = "pending" if j["attempts"] < int(j.get("max_attempts", 2)) else "failed"
        if st == "failed": j["last_error"] = "supervisor died during the job twice"
        move(j, st)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exit-when-empty", action="store_true"); ap.add_argument("--no-plan", action="store_true")
    ap.add_argument("--no-legacy-wait", action="store_true"); ap.add_argument("--no-caffeinate", action="store_true")
    a = ap.parse_args()
    for d in Q.values(): d.mkdir(parents=True, exist_ok=True)
    (RUNNER / "logs").mkdir(exist_ok=True)
    lock = open(RUNNER / "supervisor.lock", "w")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another supervisor holds the lock; exiting"); time.sleep(30); return 0
    (RUNNER / "supervisor.pid").write_text(str(os.getpid()))
    caf = None
    if not a.no_caffeinate and shutil.which("caffeinate"):
        caf = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    log(f"SUPERVISOR_START pid={os.getpid()} caffeinate={caf.pid if caf else None} root={RUNNER}")
    recover_running()
    def _term(*_):
        log("SUPERVISOR_STOP (signal)"); heartbeat(policy(), "stopped", ["supervisor_stopped"], {}); sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    while True:
        try:
            pol = policy()
            for f in (LOG, RUNNER / "logs" / "launchd.out", RUNNER / "logs" / "launchd.err"): rotate(f)
            b, genv = guards(pol, check_legacy=not a.no_legacy_wait)
            pend = sorted(list_jobs("pending"), key=job_sort_key)
            if b:
                heartbeat(pol, "waiting" if any(x.startswith("waiting_legacy") for x in b) else "paused", b, genv); time.sleep(30); continue
            if not pend:
                if a.exit_when_empty and (a.no_plan or (RUNNER / "NOPLAN").exists()):
                    heartbeat(pol, "idle", [], genv); log("EXIT (queue empty)"); return 0
                if a.no_plan or (RUNNER / "NOPLAN").exists():
                    heartbeat(pol, "idle", ["noplan"], genv); time.sleep(30); continue
                jobs = planner.plan()
                STATE["last_plan"] = {"at": now_iso(), "jobs": [j["id"] for j in jobs]}
                log(f"PLAN {[j['id'] for j in jobs]}")
                if not jobs:
                    heartbeat(pol, "idle", ["planner_exhausted"], genv); time.sleep(300)
                continue
            heartbeat(pol, "starting", [], genv)
            run_job(pend[0], pol, check_legacy=not a.no_legacy_wait)
        except SystemExit:
            raise
        except Exception:
            log("ERROR " + traceback.format_exc().replace("\n", " | ")); STATE["current"] = None
            try: recover_running()
            except Exception: pass
            time.sleep(60)

if __name__ == "__main__":
    sys.exit(main() or 0)
