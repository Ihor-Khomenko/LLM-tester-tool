"""Run directory layout: config.yaml, env.json, results.jsonl.gz, metrics.json, README."""

from __future__ import annotations

import csv
import gzip
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, TextIO, Union

from llm_tester.artifacts.env_info import collect_env
from llm_tester.artifacts.secrets import argv_without_secrets, redact_mapping, scrub_secrets

try:
    import yaml
except ImportError:  # pragma: no cover — optional until [sentinel]/dev]
    yaml = None  # type: ignore


RESULTS_NAME = "results.jsonl.gz"
CONFIG_NAME = "config.yaml"
ENV_NAME = "env.json"
METRICS_NAME = "metrics.json"
README_NAME = "README.md"


def _slug(text: str, *, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower()).strip("-")
    return (s or "run")[:max_len]


def default_run_dir(
    command: str,
    *,
    name: Optional[str] = None,
    root: Optional[Path] = None,
    when: Optional[datetime] = None,
) -> Path:
    """Default path: ``runs/<YYYYMMDD>_<name>`` under ``root`` (cwd)."""
    root = root or Path.cwd()
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%d")
    label = _slug(name or command)
    return root / "runs" / f"{ts}_{label}"


def resolve_run_dir(
    run_dir: Optional[Union[str, Path]],
    command: str,
    *,
    name: Optional[str] = None,
    root: Optional[Path] = None,
) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    return default_run_dir(command, name=name, root=root)


@dataclass
class RunArtifactBundle:
    """Paths for one reproducible experiment run."""

    path: Path
    command: str = "run"
    config: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def results_path(self) -> Path:
        return self.path / RESULTS_NAME

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_NAME

    @property
    def env_path(self) -> Path:
        return self.path / ENV_NAME

    @property
    def metrics_path(self) -> Path:
        return self.path / METRICS_NAME

    @property
    def readme_path(self) -> Path:
        return self.path / README_NAME

    def ensure(self) -> "RunArtifactBundle":
        self.path.mkdir(parents=True, exist_ok=True)
        return self


class GzipJSONLWriter:
    """Append-friendly gzip JSONL (concatenated gzip members)."""

    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ab" if append and self.path.exists() else "wb"
        self._fh = gzip.open(self.path, mode)

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(scrub_secrets(record), ensure_ascii=False) + "\n"
        self._fh.write(line.encode("utf-8"))
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "GzipJSONLWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def open_results_writer(run_dir: Path, *, append: bool = False) -> GzipJSONLWriter:
    return GzipJSONLWriter(Path(run_dir) / RESULTS_NAME, append=append)


def load_results(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load records from results.jsonl.gz or plain .jsonl / legacy .json."""
    p = Path(path)
    if p.is_dir():
        p = p / RESULTS_NAME
    if not p.exists():
        raise FileNotFoundError(f"Results not found: {p}")

    rows: List[Dict[str, Any]] = []
    if p.suffix == ".gz" or p.name.endswith(".jsonl.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    if p.suffix == ".jsonl" or "\n" in text and text.lstrip().startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    data = json.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return rows


def load_completed_ids_gz(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    try:
        for rec in load_results(path):
            pid = rec.get("id") or rec.get("scenario_id") or rec.get("trace_id")
            if pid is not None:
                done.add(str(pid))
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile):
        pass
    return done


def _dump_yaml(data: Dict[str, Any], path: Path) -> None:
    clean = redact_mapping(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(clean, fh, sort_keys=False, allow_unicode=True)
        return
    # Fallback without PyYAML: JSON-compatible YAML subset.
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# YAML fallback (install PyYAML for pretty output)\n")
        json.dump(clean, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def write_config(
    run_dir: Path,
    config: Dict[str, Any],
    *,
    argv: Optional[Sequence[str]] = None,
) -> Path:
    payload = dict(config)
    if argv is not None:
        payload["argv"] = argv_without_secrets(argv)
    path = Path(run_dir) / CONFIG_NAME
    _dump_yaml(payload, path)
    return path


def write_env(run_dir: Path, env: Dict[str, Any]) -> Path:
    path = Path(run_dir) / ENV_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(scrub_secrets(env), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def write_metrics(run_dir: Path, metrics: Dict[str, Any]) -> Path:
    path = Path(run_dir) / METRICS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(scrub_secrets(metrics), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def load_metrics(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / METRICS_NAME
    if not p.exists():
        raise FileNotFoundError(f"metrics.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / CONFIG_NAME
    if not p.exists():
        raise FileNotFoundError(f"config.yaml not found: {p}")
    text = p.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    # Fallback: try JSON
    try:
        # Skip comment lines
        body = "\n".join(
            ln for ln in text.splitlines() if not ln.strip().startswith("#")
        )
        data = json.loads(body)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def write_run_readme(
    run_dir: Path,
    *,
    command: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    env: Optional[Dict[str, Any]] = None,
) -> Path:
    path = Path(run_dir) / README_NAME
    sha = (env or {}).get("git", {}).get("sha") if env else None
    lines = [
        f"# Run artefact: `{Path(run_dir).name}`",
        "",
        f"- **Command:** `{command}`",
        f"- **Created (UTC):** {(env or {}).get('timestamp_utc', 'n/a')}",
        f"- **Git SHA:** `{sha or 'n/a'}`",
        f"- **Provider / model:** "
        f"`{(env or {}).get('provider')}` / `{(env or {}).get('model')}`",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `config.yaml` | Exact CLI args, params, thresholds (secrets redacted) |",
        "| `env.json` | Git SHA, package versions, hardware, timestamp |",
        "| `results.jsonl.gz` | Per-sample results (gzip JSONL) |",
        "| `metrics.json` | Aggregate metrics |",
        "| `README.md` | This summary |",
        "",
        "## Config (summary)",
        "",
        "```yaml",
        _brief_yaml(redact_mapping(config)),
        "```",
        "",
        "## Metrics (summary)",
        "",
        "```json",
        json.dumps(_metrics_preview(metrics), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Thesis note",
        "",
        "Copy this directory under `experiments/runs/` in the dissertation repo. "
        "Figures: `llmtest figures <this-dir> --out figures/`. "
        "Black-and-white, max width ≈158 mm (A4 with 30/15/20/20 mm margins).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _brief_yaml(data: Dict[str, Any], *, max_keys: int = 40) -> str:
    keys = list(data.keys())[:max_keys]
    subset = {k: data[k] for k in keys}
    if yaml is not None:
        return yaml.safe_dump(subset, sort_keys=False, allow_unicode=True).rstrip()
    return json.dumps(subset, indent=2, ensure_ascii=False)


def _metrics_preview(metrics: Dict[str, Any]) -> Dict[str, Any]:
    skip = {"latencies", "latencies_ms", "by_attack_type", "by_provider_model"}
    out = {k: v for k, v in metrics.items() if k not in skip}
    if "by_attack_type" in metrics:
        out["by_attack_type_keys"] = sorted(metrics["by_attack_type"].keys())
    return out


def finalize_run_dir(
    run_dir: Path,
    *,
    command: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    argv: Optional[Sequence[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    extra_sha: Optional[Sequence[str]] = None,
    env_extra: Optional[Dict[str, Any]] = None,
) -> RunArtifactBundle:
    """Write config/env/metrics/README into an existing run directory."""
    bundle = RunArtifactBundle(path=Path(run_dir), command=command).ensure()
    write_config(bundle.path, config, argv=argv)
    env = collect_env(
        provider=provider,
        model=model,
        extra_sha=extra_sha,
        command=command,
        extra=env_extra,
    )
    write_env(bundle.path, env)
    write_metrics(bundle.path, metrics)
    write_run_readme(
        bundle.path,
        command=command,
        config=config,
        metrics=metrics,
        env=env,
    )
    bundle.config = config
    bundle.env = env
    bundle.metrics = metrics
    return bundle


def is_run_dir(path: Path) -> bool:
    p = Path(path)
    return p.is_dir() and (p / METRICS_NAME).exists()


def iter_run_dirs(runs_root: Path) -> List[Path]:
    root = Path(runs_root)
    if is_run_dir(root):
        return [root]
    if not root.is_dir():
        return []
    found = [p for p in sorted(root.iterdir()) if is_run_dir(p)]
    return found
