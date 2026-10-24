"""JSONL decision logging for MCP-Sentinel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

from llm_tester.sentinel.types import GuardDecision


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionLogger:
    def __init__(self, path: Optional[Path] = None, *, append: bool = True) -> None:
        self.path: Optional[Path] = Path(path) if path else None
        self._fh: Optional[TextIO] = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a" if append else "w", encoding="utf-8")

    def log(self, decision: GuardDecision, **extra: Any) -> Dict[str, Any]:
        record = decision.to_dict()
        record["timestamp"] = _utc_now()
        record.update(extra)
        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
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
