#!/bin/bash
# Oberih runner control. Usage: ctl.sh status|pause|resume|stop-now|enqueue <file.json|'{json}'>|drain|undrain|plan [N]|analyze|logs|load|unload|restart|selftest
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${OBERIH_TESTS_ROOT:-${RESEARCH_ROOT:-$(dirname "$R")}}"
export OBERIH_RUNNER_ROOT="$R" OBERIH_TESTS_ROOT="$ROOT" RESEARCH_ROOT="$ROOT"

# Resolve interpreter + llm_tester PYTHONPATH from policy.yaml / env (Mac defaults kept in policy).
if [ -n "${OBERIH_VENV_PYTHON:-${OBERIH_PYTHON:-}}" ] && [ -x "${OBERIH_VENV_PYTHON:-${OBERIH_PYTHON}}" ]; then
  PY="${OBERIH_VENV_PYTHON:-$OBERIH_PYTHON}"
else
  VENV_PY="$(python3 -c "import os,sys; sys.path.insert(0,r'$R/lib'); from common import venv_python; print(venv_python() or '')" 2>/dev/null || true)"
  if [ -n "$VENV_PY" ] && [ -x "$VENV_PY" ]; then PY="$VENV_PY"; else PY=python3; fi
fi
if [ -n "${LLM_TESTER_ROOT:-}" ]; then
  export PYTHONPATH="$LLM_TESTER_ROOT"
else
  LT="$("$PY" -c "import sys; sys.path.insert(0,r'$R/lib'); from common import llm_tester_root; print(llm_tester_root())" 2>/dev/null || echo "$ROOT/LLM-tester-tool-fafd500")"
  export PYTHONPATH="$LT"
fi

LABEL="${OBERIH_LAUNCHD_LABEL:-ua.ihkh.oberih-runner}"
PLIST="${OBERIH_LAUNCHD_PLIST:-$HOME/Library/LaunchAgents/$LABEL.plist}"
cmd="${1:-status}"; shift || true
case "$cmd" in
  status)
    "$PY" - "$R" <<'PY'
import json,sys,os,time
R=sys.argv[1]; s=json.load(open(R+"/status.json")) if os.path.exists(R+"/status.json") else {}
age=time.time()-os.path.getmtime(R+"/status.json") if s else None
print(f"state={s.get('state')} blockers={s.get('blockers')} heartbeat_age_s={age and round(age)} pid={s.get('pid')}")
print("guards:", s.get("guard_state"), "queue:", s.get("queue"), "loaded:", s.get("ollama_loaded"))
print("current:", s.get("current")); print("last_finished:", s.get("last_finished")); print("last_plan:", s.get("last_plan"))
PY
    if command -v launchctl >/dev/null 2>&1; then
      launchctl list | grep -i oberih || echo "(LaunchAgent not loaded)"
    else
      echo "(launchctl not available on this OS)"
    fi
    [ -f "$R/PAUSE" ] && echo "PAUSE file present"; [ -f "$R/NOPLAN" ] && echo "NOPLAN (drained) present"
    tail -n 5 "$R/logs/supervisor.log" 2>/dev/null ;;
  pause)   touch "$R/PAUSE"; echo "paused: current job finishes, no new jobs start (ctl.sh stop-now to also preempt the current job)";;
  resume)  rm -f "$R/PAUSE" "$R/STOPNOW"; echo resumed;;
  stop-now) touch "$R/PAUSE" "$R/STOPNOW"; echo "pausing + preempting current job (it goes back to pending and resumes later)";;
  enqueue)
    src="${1:?json file or inline json}"
    "$PY" - "$R" "$src" <<'PY'
import json,sys,os
sys.path.insert(0, sys.argv[1]+"/lib"); import planner
from common import *
s=sys.argv[2]; j=json.load(open(s)) if os.path.exists(s) else json.loads(s)
for k in ("id","kind","model","corpus"):
    assert k in j or (k in ("model","corpus") and j.get("kind")=="mock"), f"missing {k}"
base=planner.job(j["id"], j.get("category","manual"), kind=j["kind"], **{k:v for k,v in j.items() if k not in ("id","category","kind")})
base["source"]="manual"; base.setdefault("priority", 20)
assert base["id"] not in all_job_ids(), "job id exists"
write_json_atomic(Q["pending"]/f"{base['id']}.json", base); print("enqueued", base["id"], "est_min", base["est_min"])
PY
    ;;
  drain)   touch "$R/NOPLAN"; mkdir -p "$R/queue/drained"; mv "$R"/queue/pending/*.json "$R/queue/drained/" 2>/dev/null; echo "drained: pending moved to queue/drained, auto-planner off; current job finishes";;
  undrain) rm -f "$R/NOPLAN"; mv "$R"/queue/drained/*.json "$R/queue/pending/" 2>/dev/null; echo "undrained";;
  plan)    cd "$R" && "$PY" lib/planner.py --dry "${1:-10}";;
  analyze) cd "$R" && "$PY" lib/analyzer.py;;
  logs)    tail -n "${1:-40}" "$R/logs/supervisor.log";;
  load)
    if ! command -v launchctl >/dev/null 2>&1; then echo "launchctl not available"; exit 1; fi
    launchctl bootstrap "gui/$(id -u)" "$PLIST" && echo loaded;;
  unload)
    if ! command -v launchctl >/dev/null 2>&1; then echo "launchctl not available"; exit 1; fi
    launchctl bootout "gui/$(id -u)/$LABEL" && echo "unloaded (stops the runner; running job is requeued on next start)";;
  restart)
    if ! command -v launchctl >/dev/null 2>&1; then echo "launchctl not available"; exit 1; fi
    launchctl kickstart -k "gui/$(id -u)/$LABEL" && echo restarted;;
  selftest)
    T="$R/selftest"; rm -rf "$T/root" "$T/runs_auto"; mkdir -p "$T/root/queue"/{pending,running,done,failed} "$T/root/lib"
    "$PY" -c "import yaml,sys;p=yaml.safe_load(open('$R/policy.yaml'));p['legacy_wait']=[];p['guards']['require_ollama']=False;p['guards']['battery_min_pct']=0;yaml.safe_dump(p,open('$T/root/policy.yaml','w'))"
    cp "$R"/lib/*.py "$T/root/lib/"
    export OBERIH_RUNNER_ROOT="$T/root" OBERIH_TESTS_ROOT="$ROOT" OBERIH_RUNS_AUTO="$T/runs_auto"
    "$PY" - <<'PY'
import sys,os; sys.path.insert(0, os.environ["OBERIH_RUNNER_ROOT"]+"/lib"); import planner
from common import *
for jid,fail in (("selftest-ok",False),("selftest-fail",True)):
    j=planner.job(jid,"selftest",kind="mock",model="mock",corpus="std",mock_fail=fail,timeout_s=60); write_json_atomic(Q["pending"]/f"{jid}.json", j)
PY
    "$PY" "$T/root/lib/supervisor.py" --exit-when-empty --no-plan --no-legacy-wait --no-caffeinate
    echo "--- selftest result"; ls "$T/root/queue/done" "$T/root/queue/failed"; grep -E "START|FINISH|EXIT" "$T/root/logs/supervisor.log"
    "$PY" -c "import json;s=json.load(open('$T/root/status.json'));print('status.json state=',s['state'],'queue=',s['queue'])"
    ls -R "$T/runs_auto" | head -20 ;;
  *) echo "usage: ctl.sh status|pause|resume|stop-now|enqueue <json>|drain|undrain|plan [N]|analyze|logs [N]|load|unload|restart|selftest"; exit 2;;
esac
