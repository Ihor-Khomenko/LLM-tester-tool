"""
``llmtest import-mcp-sim`` — convert mcp-security-research scenario exports
into ``mcp-sentinel-trace/v1`` cases for ``guard-eval``.

Input is schema ``1.0`` JSONL from ``python -m scenarios.export`` (or the
bundled fixture). Optionally run that exporter from a local research checkout
via ``--research-root``.

Checkpoint mapping (also written under ``meta.checkpoints`` per case):

* **T0** — ``tools_listed`` / ``tools`` (and ``tools_mutated`` / ``tools_after``
  for rug-pull) → ``check_tool_registration``
* **T1** — ``tool_result`` events (untrusted observations; injection source) →
  ``on_tool_output``
* **T2** — ``tool_call`` events (arguments; ``attack_step`` preferred) →
  ``check_tool_call``
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_tester.agent.mcpsr import (
    annotate_checkpoints,
    convert_mcpsr_record,
    is_mcpsr_record,
)

SCHEMA_OUT = "mcp-sentinel-trace/v1"


def load_mcpsr_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load schema 1.0 (or already-v1) JSONL; skip blank/# lines."""
    records: List[Dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{i}: expected a JSON object")
        records.append(obj)
    return records


def convert_record(
    data: Dict[str, Any], *, default_id: str = "", line_no: int = 0
) -> Dict[str, Any]:
    """
    Convert one MSR 1.0 record to v1 with checkpoint annotations.

    Already-v1 records are passed through after annotation refresh.
    """
    if is_mcpsr_record(data):
        out = convert_mcpsr_record(data, default_id=default_id or f"line-{line_no}")
    else:
        # Accept native v1 so re-import is idempotent.
        out = dict(data)
        if not out.get("schema_version"):
            out["schema_version"] = SCHEMA_OUT
        if default_id and not out.get("id"):
            out["id"] = default_id
    return annotate_checkpoints(out, source_record=data if is_mcpsr_record(data) else None)


def convert_records(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convert many records; return (v1_cases, summary)."""
    out: List[Dict[str, Any]] = []
    n_attack = 0
    n_benign = 0
    by_checkpoint = {"T0": 0, "T1": 0, "T2": 0}
    for i, rec in enumerate(records, start=1):
        case = convert_record(rec, default_id=str(rec.get("id") or f"import-{i}"), line_no=i)
        out.append(case)
        label = str(case.get("label") or "").lower()
        if label == "benign":
            n_benign += 1
        else:
            n_attack += 1
        cps = (case.get("meta") or {}).get("checkpoints") or {}
        for key in ("T0", "T1", "T2"):
            if cps.get(key):
                by_checkpoint[key] += 1
    summary = {
        "n": len(out),
        "n_attack": n_attack,
        "n_benign": n_benign,
        "by_checkpoint": by_checkpoint,
        "schema_out": SCHEMA_OUT,
    }
    return out, summary


def write_v1_jsonl(cases: Iterable[Dict[str, Any]], path: Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_from_research_root(
    research_root: Path,
    *,
    lang: str = "en",
    scenario_ids: Optional[Sequence[str]] = None,
) -> Path:
    """
    Run ``python -m scenarios.export`` in a local mcp-security-research checkout.

    Returns the path to a temporary JSONL file (caller should delete when done).
    """
    root = Path(research_root).resolve()
    if not (root / "scenarios" / "export.py").is_file():
        raise FileNotFoundError(
            f"{root} does not look like mcp-security-research "
            "(missing scenarios/export.py)"
        )
    tmp = tempfile.NamedTemporaryFile(
        prefix="mcp-sim-export-", suffix=".jsonl", delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    cmd = [
        sys.executable,
        "-m",
        "scenarios.export",
        "--out",
        str(tmp_path),
        "--lang",
        lang,
    ]
    if scenario_ids:
        for sid in scenario_ids:
            cmd.extend(["--id", sid])
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"scenarios.export failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    return tmp_path


def import_mcp_sim(
    *,
    input_path: Optional[Path] = None,
    research_root: Optional[Path] = None,
    out_path: Path,
    lang: str = "en",
    scenario_ids: Optional[Sequence[str]] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convert MSR scenarios → v1 JSONL. Exactly one of ``input_path`` /
    ``research_root`` must be set (or both: research export then optional
    ``input_path`` is ignored when research_root is set).
    """
    tmp_export: Optional[Path] = None
    try:
        if research_root is not None:
            tmp_export = export_from_research_root(
                research_root, lang=lang, scenario_ids=scenario_ids
            )
            src = tmp_export
            source_kind = "research_export"
        elif input_path is not None:
            src = Path(input_path)
            source_kind = "jsonl"
        else:
            raise ValueError("Provide --input JSONL or --research-root")

        records = load_mcpsr_jsonl(src)
        if scenario_ids and research_root is None:
            want = set(scenario_ids)
            records = [r for r in records if str(r.get("id") or "") in want]
        cases, summary = convert_records(records)
        n_written = write_v1_jsonl(cases, out_path)
        manifest: Dict[str, Any] = {
            "command": "import-mcp-sim",
            "source_kind": source_kind,
            "source": str(Path(src).resolve()),
            "research_root": str(Path(research_root).resolve()) if research_root else None,
            "out_path": str(Path(out_path).resolve()),
            "n_written": n_written,
            **summary,
            "mapping": {
                "T0": "tools / tools_listed (+ tools_after / tools_mutated) → check_tool_registration",
                "T1": "tool_result (injection source) → on_tool_output",
                "T2": "tool_call (+ arguments; attack_step preferred) → check_tool_call",
            },
        }
        man = (
            Path(manifest_path)
            if manifest_path is not None
            else Path(out_path).with_name(Path(out_path).stem + "_import_manifest.json")
        )
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(man.resolve())
        return manifest
    finally:
        if tmp_export is not None:
            tmp_export.unlink(missing_ok=True)
