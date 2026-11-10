"""Shared helpers for the Oberih 24/7 runner (stdlib + PyYAML only). No secrets are read or written."""
import json, math, os, re, shutil, subprocess, time, datetime, urllib.request
from pathlib import Path

RUNNER = Path(os.environ.get("OBERIH_RUNNER_ROOT") or Path(__file__).resolve().parents[1])
# Research workspace (corpora, real_mcp/, runs_auto/). Defaults to parent of runner/
# (Mac: .../oberih-tests; in-repo: .../LLM-tester-tool/research).
ROOT = Path(os.environ.get("OBERIH_TESTS_ROOT") or os.environ.get("RESEARCH_ROOT") or RUNNER.parent)
RUNS_AUTO = Path(os.environ.get("OBERIH_RUNS_AUTO") or ROOT / "runs_auto")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
Q = {k: RUNNER / "queue" / k for k in ("pending", "running", "done", "failed")}


def resolve_under_root(rel_or_abs):
    """Resolve a policy path: absolute stays; relative is under ROOT."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (ROOT / p).resolve()


def llm_tester_root(pol=None):
    """Checkout / editable install that provides ``llm_tester`` (on PYTHONPATH)."""
    pol = pol or policy()
    override = os.environ.get("LLM_TESTER_ROOT") or os.environ.get("OBERIH_LLM_TESTER_ROOT")
    if override:
        return Path(override).resolve()
    return resolve_under_root(pol.get("llm_tester_dir", "LLM-tester-tool-fafd500"))


def venv_python(pol=None):
    pol = pol or policy()
    override = os.environ.get("OBERIH_VENV_PYTHON") or os.environ.get("OBERIH_PYTHON")
    if override and Path(override).exists():
        return str(Path(override))
    cand = resolve_under_root(pol.get("venv", "venv-fafd500")) / "bin" / "python"
    return str(cand) if cand.exists() else None


def default_path_env():
    """PATH for child jobs. Prefer env, then common Mac Homebrew prefixes, then current PATH."""
    if os.environ.get("OBERIH_PATH") or os.environ.get("RUNNER_PATH"):
        return os.environ.get("OBERIH_PATH") or os.environ.get("RUNNER_PATH")
    parts = []
    for p in (
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ):
        if Path(p).exists() and p not in parts:
            parts.append(p)
    for p in (os.environ.get("PATH") or "").split(":"):
        if p and p not in parts:
            parts.append(p)
    return ":".join(parts) or "/usr/bin:/bin"


def ollama_bin():
    """Resolve ``ollama`` CLI (``OLLAMA_BIN``, then PATH, then Homebrew fallbacks)."""
    for c in (
        os.environ.get("OLLAMA_BIN"),
        shutil.which("ollama"),
        "/opt/homebrew/bin/ollama",
        "/usr/local/bin/ollama",
    ):
        if c and Path(c).exists():
            return c
    return "ollama"

def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def today():
    return datetime.date.today().isoformat()

def load_yaml(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f) or {}

def policy():
    return load_yaml(RUNNER / "policy.yaml")

def read_json(p, default=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default

def write_json_atomic(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)

def write_text_atomic(p, text):
    p = Path(p); tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, p)

def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def fmt(k, n):
    if not n:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100*k/n:.0f}% [{100*lo:.0f}–{100*hi:.0f}]"

def safe(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))

# ---------------- guards ----------------
def free_gb():
    return shutil.disk_usage("/").free / 1024**3

def swap_used_mb():
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)M", out)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0

def battery():
    """(source, percent) e.g. ('AC', 100) / ('Battery', 62). Unknown -> ('AC', None)."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ("AC", None)
    src = "Battery" if "Battery Power" in out else "AC"
    m = re.search(r"(\d+)%", out)
    return (src, int(m.group(1)) if m else None)

def pid_alive(pid, must_contain=None):
    try:
        out = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return False
    if not out:
        return False
    return (must_contain in out) if must_contain else True

# ---------------- ollama (HTTP only, localhost) ----------------
def _ollama(path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")

def ollama_up():
    try:
        _ollama("/api/version", timeout=5); return True
    except Exception:
        return False

def ollama_models():
    try:
        return {m["name"]: m for m in _ollama("/api/tags").get("models", [])}
    except Exception:
        return {}

def ollama_has(model):
    ms = ollama_models()
    return model in ms or (":" not in model and f"{model}:latest" in ms)

def ollama_digest(model):
    ms = ollama_models(); m = ms.get(model) or ms.get(f"{model}:latest") or {}
    return m.get("digest"), m.get("size")

def ollama_caps(model):
    try:
        return _ollama("/api/show", {"name": model}).get("capabilities") or []
    except Exception:
        return []

def ollama_unload_all():
    """Unload every loaded model (keep_alive 0). Never deletes models."""
    try:
        loaded = [m["name"] for m in _ollama("/api/ps").get("models", [])]
    except Exception:
        return []
    for m in loaded:
        try:
            _ollama("/api/generate", {"model": m, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
    return loaded

def ollama_loaded():
    try:
        return [m["name"] for m in _ollama("/api/ps").get("models", [])]
    except Exception:
        return None

# ---------------- job helpers ----------------
def list_jobs(state):
    out = []
    for p in sorted(Q[state].glob("*.json")):
        j = read_json(p)
        if j:
            j["_path"] = str(p); out.append(j)
    return out

def job_sort_key(j):
    return (int(j.get("priority", 50)), j.get("created", ""), j.get("id", ""))

def all_job_ids():
    ids = {}
    for st in Q:
        for p in Q[st].glob("*.json"):
            ids[p.stem] = st
    return ids
