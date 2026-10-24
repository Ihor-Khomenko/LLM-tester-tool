"""JSONL decision logging for MCP-Sentinel (plain or gzip)."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union

from llm_tester.sentinel.types import GuardDecision


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_gzip_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".gz") or name.endswith(".jsonl.gz")


class DecisionLogger:
    """Append decision records to ``.jsonl`` or ``.jsonl.gz``."""

    def __init__(self, path: Optional[Path] = None, *, append: bool = True) -> None:
        self.path: Optional[Path] = Path(path) if path else None
        self._fh: Optional[Any] = None
        self._gzip = False
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._gzip = _is_gzip_path(self.path)
            if self._gzip:
                mode = "ab" if append and self.path.exists() else "wb"
                self._fh = gzip.open(self.path, mode)
            else:
                self._fh = open(
                    self.path, "a" if append else "w", encoding="utf-8"
                )

    def log(self, decision: GuardDecision, **extra: Any) -> Dict[str, Any]:
        record = decision.to_dict()
        record["timestamp"] = _utc_now()
        record.update(extra)
        if self._fh is not None:
            line = json.dumps(record, ensure_ascii=False) + "\n"
            if self._gzip:
                self._fh.write(line.encode("utf-8"))
            else:
                self._fh.write(line)
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "DecisionLogger":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def load_decision_log(path: Union[str, Path]) -> list:
    """Load decision records from plain or gzip JSONL."""
    p = Path(path)
    rows = []
    if _is_gzip_path(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
