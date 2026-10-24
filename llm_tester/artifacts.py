"""Reproducible run artifacts: config, env, results, metrics, index, compare.

Intended layout (also used under a thesis repo as ``experiments/runs/``)::

    runs/<YYYYMMDD>_<name>/
      config.yaml       # exact CLI args, params, thresholds (no secrets)
      env.json          # git SHAs, versions, provider/model, hardware, timestamp
      results.jsonl.gz  # per-case records
      metrics.json      # aggregate metrics
      README.md         # human-readable summary

Never write API keys or secrets into these files.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RUNS_ROOT = Path("runs")

CONFIG_NAME = "config.yaml"
ENV_NAME = "env.json"
RESULTS_NAME = "results.jsonl.gz"
RESULTS_PLAIN = "results.jsonl"
METRICS_NAME = "metrics.json"
README_NAME = "README.md"
INDEX_NAME = "index.csv"

SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret|"
    r"credential|bearer|private[_-]?key|session[_-]?token)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(sk-[a-zA-Z0-9_-]{8,}|ghp_[a-zA-Z0-9]{20,}|xox[baprs]-[a-zA-Z0-9-]{10,})\b"
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


def is_secret_key(key: str) -> bool:
    return bool(SECRET_KEY_RE.search(str(key)))


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        if SECRET_VALUE_RE.search(value):
            return "<redacted>"
        # Long hex-looking tokens that look like keys
        if len(value) >= 32 and re.fullmatch(r"[A-Za-z0-9_\-./+=]+", value):
            lower = value.lower()
            if any(s in lower for s in ("key", "token", "secret")):
                return "<redacted>"
        return value
    if isinstance(value, Mapping):
        return redact_secrets(value)
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def redact_secrets(obj: Any) -> Any:
    """Recursively strip secret-looking keys/values. Never raises."""
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if is_secret_key(key):
                out[key] = "<redacted>"
            else:
                out[key] = redact_secrets(v)
        return out
    if isinstance(obj, list):
        return [redact_secrets(v) for v in obj]
    return redact_value(obj)


# ---------------------------------------------------------------------------
# YAML helpers (PyYAML when available; safe minimal fallback otherwise)
# ---------------------------------------------------------------------------


def _dump_yaml(data: Any, path: Path) -> None:
    clean = redact_secrets(data)
    try:
        import yaml

        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                clean,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        return
    except ImportError:
        pass
    # Minimal YAML emitter for dict/list/scalars used in configs.
    with path.open("w", encoding="utf-8") as fh:
        fh.write(_to_simple_yaml(clean))


def _to_simple_yaml(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent
    if obj is None:
        return "null\n"
    if isinstance(obj, bool):
        return ("true" if obj else "false") + "\n"
    if isinstance(obj, (int, float)):
        return f"{obj}\n"
    if isinstance(obj, str):
        if obj == "" or any(c in obj for c in ":#\n\"'{[]}&*!|>%@`"):
            return json.dumps(obj, ensure_ascii=False) + "\n"
        return obj + "\n"
    if isinstance(obj, list):
        if not obj:
            return "[]\n"
        lines: List[str] = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-\n{_to_simple_yaml(item, indent + 1)}")
            else:
                lines.append(f"{sp}- {_to_simple_yaml(item, 0).rstrip()}\n")
        return "".join(lines)
    if isinstance(obj, dict):
        if not obj:
            return "{}\n"
        parts = []
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{sp}{key}:\n{_to_simple_yaml(v, indent + 1)}")
            else:
                parts.append(f"{sp}{key}: {_to_simple_yaml(v, 0).rstrip()}\n")
        return "".join(parts)
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except ImportError:
        # Extremely limited fallback — prefer JSON-compatible configs.
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


# ---------------------------------------------------------------------------
# Paths / naming
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _date_stamp(when: Optional[datetime] = None) -> str:
    when = when or _utc_now()
    return when.strftime("%Y%m%d")


def sanitize_run_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._+-]+", "-", name.strip())
    cleaned = cleaned.strip("-._")
    return (cleaned or "run")[:120]


def default_run_name(
    *,
    command: str,
    dataset: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    parts = [command]
    if dataset:
        parts.append(dataset)
    if provider:
        parts.append(provider)
    if model:
        parts.append(model.replace("/", "-").replace(":", "-"))
    if extra:
        parts.append(extra)
    return sanitize_run_name("_".join(parts))


def resolve_run_dir(
    run_dir: Optional[PathLike] = None,
    *,
    run_name: Optional[str] = None,
    runs_root: PathLike = DEFAULT_RUNS_ROOT,
    command: str = "run",
    dataset: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    when: Optional[datetime] = None,
) -> Path:
    """Return the directory for this run (created by the caller)."""
    if run_dir is not None:
        return Path(run_dir)
    name = sanitize_run_name(
        run_name
        or default_run_name(
            command=command, dataset=dataset, provider=provider, model=model
        )
    )
    stamp = _date_stamp(when)
    return Path(runs_root) / f"{stamp}_{name}"


def ensure_run_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def git_sha(repo: Optional[PathLike] = None) -> Optional[str]:
    """Return HEAD SHA for ``repo`` (or CWD), or None if not a git checkout."""
    cwd = str(Path(repo) if repo is not None else Path.cwd())
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def parse_extra_sha(spec: str) -> Tuple[str, str]:
    """Parse ``name=sha`` or a filesystem path (resolves git SHA)."""
    spec = spec.strip()
    if "=" in spec and not spec.startswith("="):
        name, _, sha = spec.partition("=")
        name = name.strip()
        sha = sha.strip()
        if name and sha:
            # If RHS looks like a path, resolve it.
            path = Path(sha)
            if path.exists() and path.is_dir():
                resolved = git_sha(path)
                if not resolved:
                    raise ValueError(f"No git SHA at path for --extra-sha: {sha}")
                return name, resolved
            return name, sha
    path = Path(spec)
    if path.exists():
        resolved = git_sha(path)
        if not resolved:
            raise ValueError(f"No git SHA at path for --extra-sha: {spec}")
        return path.resolve().name, resolved
    raise ValueError(
        f"Invalid --extra-sha '{spec}'. Use name=sha or a path to a git repo."
    )


def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", None)
        except Exception:
            versions[name] = None
    # Prefer importlib.metadata for installed dists
    try:
        from importlib import metadata

        for name in names:
            try:
                versions[name] = metadata.version(name)
            except Exception:
                pass
    except Exception:
        pass
    # llm-tester package name vs module
    try:
        from importlib import metadata

        try:
            versions["llm-tester"] = metadata.version("llm-tester")
        except Exception:
            versions.setdefault("llm-tester", None)
    except Exception:
        pass
    return versions


def hardware_summary() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "python_build": list(platform.python_build()),
        "cpu_count": os.cpu_count(),
    }
    try:
        if hasattr(os, "uname"):
            u = os.uname()
            info["uname"] = {
                "sysname": u.sysname,
                "nodename": u.nodename,
                "release": u.release,
                "version": u.version,
                "machine": u.machine,
            }
    except Exception:
        pass
    # Best-effort memory (Linux)
    mem_total = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    mem_total = int(parts[1]) * 1024  # kB → bytes
                    break
    except Exception:
        pass
    info["mem_total_bytes"] = mem_total
    return info


def collect_env(
    *,
    command: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    extra_shas: Optional[Sequence[Tuple[str, str]]] = None,
    repo: Optional[PathLike] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build env.json payload (already safe — no secrets)."""
    ts = timestamp or _utc_now().isoformat()
    pkgs = package_versions(
        [
            "openai",
            "datasets",
            "dotenv",
            "yaml",
            "llm_guard",
            "sentence_transformers",
            "mcp",
        ]
    )
    # Normalize dotenv / yaml import names to package names when possible
    try:
        from importlib import metadata

        for dist, key in (
            ("python-dotenv", "python-dotenv"),
            ("PyYAML", "PyYAML"),
            ("llm-guard", "llm-guard"),
            ("sentence-transformers", "sentence-transformers"),
        ):
            try:
                pkgs[key] = metadata.version(dist)
            except Exception:
                pkgs.setdefault(key, None)
    except Exception:
        pass

    env: Dict[str, Any] = {
        "timestamp": ts,
        "command": command,
        "git": {
            "llm_tester_sha": git_sha(repo),
            "extra": {name: sha for name, sha in (extra_shas or [])},
        },
        "python": {
            "version": sys.version.split()[0],
            "version_info": list(sys.version_info[:3]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": pkgs,
        "provider": provider,
        "model": model,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "hardware": hardware_summary(),
    }
    return redact_secrets(env)


# ---------------------------------------------------------------------------
# Results gzip I/O
# ---------------------------------------------------------------------------


def write_jsonl_gz(path: PathLike, records: Iterable[Mapping[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(redact_secrets(dict(rec)), ensure_ascii=False) + "\n")
            n += 1
    return n


def ensure_plain_jsonl_for_resume(run_dir: PathLike, out_path: Optional[PathLike] = None) -> Path:
    """If only ``results.jsonl.gz`` exists, decompress it so ``--resume`` can append."""
    run_dir = Path(run_dir)
    out = Path(out_path) if out_path is not None else run_dir / RESULTS_PLAIN
    gz = run_dir / RESULTS_NAME
    if out.exists():
        return out
    if gz.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(gz, "rt", encoding="utf-8") as fin, out.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                fout.write(line if line.endswith("\n") else line + "\n")
    return out


def gzip_jsonl_file(src: PathLike, dest: Optional[PathLike] = None, *, remove_src: bool = False) -> Path:
    """Compress a plain JSONL file to ``results.jsonl.gz`` (redacting line-by-line)."""
    src = Path(src)
    dest = Path(dest) if dest is not None else src.with_suffix(src.suffix + ".gz")
    if src.suffixes[-2:] == [".jsonl", ".gz"] or str(src).endswith(".jsonl.gz"):
        # Already gzipped — copy if needed
        if Path(dest) != src:
            shutil.copy2(src, dest)
        return Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, gzip.open(dest, "wt", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                fout.write(json.dumps(redact_secrets(obj), ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                fout.write(line + "\n")
    if remove_src and src != dest and src.exists():
        src.unlink()
    return Path(dest)


def iter_jsonl_records(path: PathLike) -> Iterable[Dict[str, Any]]:
    """Yield dict records from ``.jsonl`` or ``.jsonl.gz`` (or legacy ``.json``)."""
    path = Path(path)
    if not path.exists():
        return
    opener = gzip.open if str(path).endswith(".gz") else open
    mode = "rt" if str(path).endswith(".gz") else "r"
    # Heuristic for plain JSON array
    if path.suffix == ".json" and not str(path).endswith(".jsonl"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        elif isinstance(data, dict):
            yield data
        return
    with opener(path, mode, encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def results_path_in_run(run_dir: PathLike) -> Path:
    run_dir = Path(run_dir)
    gz = run_dir / RESULTS_NAME
    plain = run_dir / RESULTS_PLAIN
    if gz.exists():
        return gz
    if plain.exists():
        return plain
    return gz


def load_completed_ids_from_results(path: PathLike, id_keys: Sequence[str] = ("id", "scenario_id")) -> set:
    done: set = set()
    path = Path(path)
    if not path.exists():
        # Also try sibling .gz / plain
        candidates = [path]
        if str(path).endswith(".jsonl.gz"):
            candidates.append(Path(str(path)[:-3]))
        elif str(path).endswith(".jsonl"):
            candidates.append(Path(str(path) + ".gz"))
        for c in candidates:
            if c.exists():
                path = c
                break
        else:
            return done
    for rec in iter_jsonl_records(path):
        for key in id_keys:
            if rec.get(key) is not None:
                done.add(str(rec[key]))
                break
    return done


# ---------------------------------------------------------------------------
# Writing a full run directory
# ---------------------------------------------------------------------------


@dataclass
class RunArtifacts:
    run_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    results_path: Optional[Path] = None

    @property
    def name(self) -> str:
        return self.run_dir.name


def write_config(run_dir: PathLike, config: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / CONFIG_NAME
    payload = dict(config)
    payload.setdefault("schema", "llmtest-run-config/v1")
    _dump_yaml(payload, path)
    return path


def write_env(run_dir: PathLike, env: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / ENV_NAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(redact_secrets(dict(env)), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def write_metrics(run_dir: PathLike, metrics: Mapping[str, Any]) -> Path:
    path = Path(run_dir) / METRICS_NAME
    with path.open("w", encoding="utf-8") as fh:
        json.dump(redact_secrets(dict(metrics)), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def _fmt_metric(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if 0.0 <= v <= 1.0:
            return f"{v:.4f}"
        return f"{v:.4g}"
    return str(v)


def write_readme(
    run_dir: PathLike,
    *,
    config: Mapping[str, Any],
    env: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    path = Path(run_dir) / README_NAME
    command = config.get("command") or env.get("command") or "run"
    lines = [
        f"# Run: {Path(run_dir).name}",
        "",
        f"- **Command:** `{command}`",
        f"- **Timestamp:** {env.get('timestamp', 'n/a')}",
        f"- **Provider / model:** {env.get('provider') or config.get('provider')} / "
        f"{env.get('model') or config.get('model')}",
        f"- **Git SHA (llm_tester):** `{(env.get('git') or {}).get('llm_tester_sha') or 'n/a'}`",
    ]
    extra = (env.get("git") or {}).get("extra") or {}
    if extra:
        for name, sha in extra.items():
            lines.append(f"- **Git SHA ({name}):** `{sha}`")
    lines.extend(["", "## Metrics", ""])
    # Prefer flat numeric metrics
    skip = {"by_provider_model", "by_attack_type", "files", "latencies"}
    for key, val in metrics.items():
        if key in skip or isinstance(val, (dict, list)):
            continue
        lines.append(f"- **{key}:** {_fmt_metric(val)}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{CONFIG_NAME}` — exact args, params, thresholds",
            f"- `{ENV_NAME}` — environment fingerprint (no secrets)",
            f"- `{RESULTS_NAME}` — per-case results",
            f"- `{METRICS_NAME}` — aggregate metrics",
            "",
            "Copy this directory into a thesis repo under `experiments/runs/` for archival.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def finalize_run(
    run_dir: PathLike,
    *,
    config: Mapping[str, Any],
    env: Mapping[str, Any],
    metrics: Mapping[str, Any],
    results_jsonl: Optional[PathLike] = None,
    results_records: Optional[Iterable[Mapping[str, Any]]] = None,
    remove_plain_results: bool = True,
) -> RunArtifacts:
    """Write all standard artifacts into ``run_dir``."""
    run_dir = ensure_run_dir(run_dir)
    write_config(run_dir, config)
    write_env(run_dir, env)
    write_metrics(run_dir, metrics)
    results_dest = run_dir / RESULTS_NAME
    if results_records is not None:
        write_jsonl_gz(results_dest, results_records)
    elif results_jsonl is not None:
        src = Path(results_jsonl)
        if src.exists():
            if src.resolve() == results_dest.resolve():
                pass
            elif str(src).endswith(".gz"):
                if src.resolve() != results_dest.resolve():
                    shutil.copy2(src, results_dest)
            else:
                gzip_jsonl_file(src, results_dest, remove_src=remove_plain_results and src.parent == run_dir)
        else:
            write_jsonl_gz(results_dest, [])
    elif not results_dest.exists():
        write_jsonl_gz(results_dest, [])
    write_readme(run_dir, config=config, env=env, metrics=metrics)
    return RunArtifacts(
        run_dir=run_dir,
        config=dict(config),
        env=dict(env),
        metrics=dict(metrics),
        results_path=results_dest if results_dest.exists() else None,
    )


def load_run(run_dir: PathLike) -> RunArtifacts:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    config: Dict[str, Any] = {}
    env: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    cfg_path = run_dir / CONFIG_NAME
    if cfg_path.exists():
        config = _load_yaml(cfg_path)
    env_path = run_dir / ENV_NAME
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
    metrics_path = run_dir / METRICS_NAME
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return RunArtifacts(
        run_dir=run_dir,
        config=config,
        env=env,
        metrics=metrics,
        results_path=results_path_in_run(run_dir) if (run_dir / RESULTS_NAME).exists() or (run_dir / RESULTS_PLAIN).exists() else None,
    )


def is_run_dir(path: PathLike) -> bool:
    p = Path(path)
    return p.is_dir() and (p / METRICS_NAME).exists() and (p / CONFIG_NAME).exists()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

INDEX_FIELDS = [
    "run_dir",
    "name",
    "command",
    "timestamp",
    "provider",
    "model",
    "dataset",
    "guard",
    "git_sha",
    "n",
    "refusal_rate",
    "attack_success_rate",
    "tpr",
    "fpr",
    "f1",
    "asr_reduction",
    "mean_latency_s",
    "latency_p50_ms",
]


def _index_row(art: RunArtifacts) -> Dict[str, Any]:
    m = art.metrics
    c = art.config
    e = art.env
    return {
        "run_dir": str(art.run_dir),
        "name": art.name,
        "command": c.get("command") or e.get("command") or "",
        "timestamp": e.get("timestamp") or "",
        "provider": e.get("provider") or c.get("provider") or "",
        "model": e.get("model") or c.get("model") or "",
        "dataset": c.get("dataset") or "",
        "guard": c.get("guard") or "",
        "git_sha": (e.get("git") or {}).get("llm_tester_sha") or "",
        "n": m.get("n", m.get("total", m.get("evaluated", ""))),
        "refusal_rate": m.get("refusal_rate", ""),
        "attack_success_rate": m.get("attack_success_rate", ""),
        "tpr": m.get("tpr", ""),
        "fpr": m.get("fpr", ""),
        "f1": m.get("f1", ""),
        "asr_reduction": m.get("asr_reduction", ""),
        "mean_latency_s": m.get("mean_latency_s", ""),
        "latency_p50_ms": m.get("latency_p50_ms", ""),
    }


def discover_runs(runs_root: PathLike) -> List[Path]:
    root = Path(runs_root)
    if not root.exists():
        return []
    if is_run_dir(root):
        return [root]
    runs = [p for p in sorted(root.iterdir()) if is_run_dir(p)]
    return runs


def write_index(runs_root: PathLike, out: Optional[PathLike] = None) -> Path:
    root = Path(runs_root)
    out_path = Path(out) if out is not None else root / INDEX_NAME
    rows = [_index_row(load_run(p)) for p in discover_runs(root)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in metrics.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, Mapping):
            # Skip deep nested buckets in the default table; keep one-level counts
            if k in ("by_provider_model", "by_attack_type"):
                continue
            flat.update(_flatten_metrics(v, key))
        elif isinstance(v, list):
            continue
        else:
            flat[key if prefix else k] = v
    return flat


def compare_metrics(
    runs: Sequence[RunArtifacts],
    *,
    keys: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return rows: metric, run0, run1, ..., delta_01, ..."""
    flats = [_flatten_metrics(r.metrics) for r in runs]
    if keys is None:
        key_set = set()
        for f in flats:
            key_set.update(f.keys())
        # Stable preference order then alpha
        preferred = [
            "n",
            "total",
            "evaluated",
            "refusal_rate",
            "attack_success_rate",
            "tpr",
            "fpr",
            "precision",
            "recall",
            "f1",
            "asr_without_guard",
            "asr_with_guard",
            "asr_reduction",
            "escalation_rate",
            "mean_latency_s",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "errors",
            "blocked",
            "refusals",
            "attack_successes",
            "tp",
            "fp",
            "tn",
            "fn",
        ]
        keys = [k for k in preferred if k in key_set] + sorted(
            k for k in key_set if k not in preferred
        )
    rows: List[Dict[str, Any]] = []
    for key in keys:
        row: Dict[str, Any] = {"metric": key}
        vals = []
        for i, flat in enumerate(flats):
            v = flat.get(key)
            row[f"run_{i}"] = v
            vals.append(v)
        # Pairwise deltas vs run_0
        base = vals[0] if vals else None
        for i, v in enumerate(vals[1:], start=1):
            if isinstance(base, (int, float)) and isinstance(v, (int, float)):
                row[f"delta_0{i}"] = v - base
            else:
                row[f"delta_0{i}"] = None
        rows.append(row)
    return rows


def format_compare_table(
    runs: Sequence[RunArtifacts],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    headers = ["metric"] + [r.name for r in runs]
    if len(runs) >= 2:
        for i in range(1, len(runs)):
            headers.append(f"Δ({runs[0].name[:12]}→{runs[i].name[:12]})")

    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if abs(v) <= 1.0:
                return f"{v:.4f}"
            return f"{v:.4g}"
        return str(v)

    # Column widths
    col_data: List[List[str]] = []
    for row in rows:
        line = [str(row.get("metric", ""))]
        for i in range(len(runs)):
            line.append(cell(row.get(f"run_{i}")))
        for i in range(1, len(runs)):
            line.append(cell(row.get(f"delta_0{i}")))
        col_data.append(line)

    widths = [len(h) for h in headers]
    for line in col_data:
        for i, c in enumerate(line):
            widths[i] = max(widths[i], len(c))

    def fmt_line(parts: List[str]) -> str:
        return " | ".join(p.ljust(widths[i]) for i, p in enumerate(parts))

    out_lines = [fmt_line(headers), "-+-".join("-" * w for w in widths)]
    for line in col_data:
        out_lines.append(fmt_line(line))
    return "\n".join(out_lines)


def format_compare_markdown(
    runs: Sequence[RunArtifacts],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    headers = ["metric"] + [r.name for r in runs]
    if len(runs) >= 2:
        for i in range(1, len(runs)):
            headers.append(f"delta_0{i}")

    def cell(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        parts = [str(row.get("metric", ""))]
        for i in range(len(runs)):
            parts.append(cell(row.get(f"run_{i}")))
        for i in range(1, len(runs)):
            parts.append(cell(row.get(f"delta_0{i}")))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def write_compare_csv(
    path: PathLike,
    runs: Sequence[RunArtifacts],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = Path(path)
    fieldnames = ["metric"] + [f"run_{i}" for i in range(len(runs))]
    fieldnames += [f"delta_0{i}" for i in range(1, len(runs))]
    # Also include human names as a comment row? Keep machine ids; add name map header via separate meta.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        # Preamble mapping run_i → directory name
        fh.write("# " + ", ".join(f"run_{i}={r.name}" for i, r in enumerate(runs)) + "\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def content_fingerprint(path: PathLike) -> str:
    """SHA256 of a file for integrity notes (optional)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
