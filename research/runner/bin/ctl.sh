#!/bin/bash
# Oberih runner control. Usage: ctl.sh status|pause|resume|stop-now|enqueue <file.json|'{json}'>|drain|undrain|plan [N]|analyze|logs|load|unload|restart|selftest
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${OBERIH_TESTS_ROOT:-${RESEARCH_ROOT:-$(dirname "$R")}}"
export OBERIH_RUNNER_ROOT="$R" OBERIH_TESTS_ROOT="$ROOT" RESEARCH_ROOT="$ROOT"

# awk-based policy field reader (no Python import required for bootstrap paths).
_policy_field() {
  # Usage: _policy_field <key>  -> first bare scalar value for top-level key
  local key="$1"
  awk -v k="$key" '
    $0 ~ "^[[:space:]]*"k":[[:space:]]*" {
      line=$0
      sub("^[[:space:]]*"k":[[:space:]]*", "", line)
      # drop inline comments
      sub(/[[:space:]]+#.*$/, "", line)
      gsub(/["'\'']/, "", line)
      # take first token (YAML scalar)
      gsub(/[[:space:]]+.*/, "", line)
      if (length(line) > 0) { print line; exit }
    }
  ' "$R/policy.yaml" 2>/dev/null || true
}

# Resolve interpreter from env, else policy.venv under workspace, else python3.
if [ -n "${OBERIH_VENV_PYTHON:-}" ] && [ -x "${OBERIH_VENV_PYTHON}" ]; then
  PY="${OBERIH_VENV_PYTHON}"
elif [ -n "${OBERIH_PYTHON:-}" ] && [ -x "${OBERIH_PYTHON}" ]; then
  PY="${OBERIH_PYTHON}"
else
  VENV_REL="$(_policy_field venv)"
  VENV_PY=""
  if [ -n "$VENV_REL" ]; then
    case "$VENV_REL" in
      /*) VENV_CAND="${VENV_REL}/bin/python" ;;
      *)  VENV_CAND="${ROOT}/${VENV_REL}/bin/python" ;;
    esac
    if [ -x "$VENV_CAND" ]; then VENV_PY="$VENV_CAND"; fi
  fi
  if [ -z "$VENV_PY" ]; then
    VENV_PY="$(python3 -c "import os,sys; sys.path.insert(0,r'$R/lib'); from common import venv_python; print(venv_python() or '')" 2>/dev/null || true)"
  fi
  if [ -n "$VENV_PY" ] && [ -x "$VENV_PY" ]; then PY="$VENV_PY"; else PY=python3; fi
fi
if [ -n "${LLM_TESTER_ROOT:-}" ]; then
  export PYTHONPATH="$LLM_TESTER_ROOT"
else
  LT_REL="$(_policy_field llm_tester_dir)"
  if [ -n "$LT_REL" ]; then
    case "$LT_REL" in
      /*) LT_DEFAULT="$LT_REL" ;;
      *)  LT_DEFAULT="$ROOT/$LT_REL" ;;
    esac
  else
    LT_DEFAULT="$ROOT/LLM-tester-tool-fafd500"
  fi
  LT="$("$PY" -c "import sys; sys.path.insert(0,r'$R/lib'); from common import llm_tester_root; print(llm_tester_root())" 2>/dev/null || echo "$LT_DEFAULT")"
  export PYTHONPATH="$LT"
fi

LABEL="${OBERIH_LAUNCHD_LABEL:-org.oberih.runner}"
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
