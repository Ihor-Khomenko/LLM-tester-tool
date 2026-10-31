"""
Shared request ledger for live research runs.

Counts HTTP requests (incl. retries) via optional httpx hooks on an OpenAI
client. Paths and caps are configurable — nothing here prints or stores API
keys. CI / ``--mock`` never installs hooks.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DAILY_CAP = 45


class BudgetExceeded(Exception):
    """Raised when a hard request cap is hit."""


def _default_ledger_path() -> Path:
    day = os.environ.get("OR_DATE") or date.today().isoformat()
    root = Path(os.environ.get("LLMTEST_LEDGER_DIR") or "runs")
    return root / f"{day}_openrouter_ledger.json"


@dataclass
class RequestLedger:
    """
    Thread-safe JSON ledger with daily + per-part caps.

    Parts typically: ``agent``, ``defense``, ``attacker``, ``s4``.

    Prefer ``reserve`` then ``commit`` around a real send so capacity is
    checked *before* the request goes out, and every request that was
    actually sent (including 429s) is persisted even if the cap is hit.
    """

    path: Path = field(default_factory=_default_ledger_path)
    daily_cap: int = field(
        default_factory=lambda: int(os.environ.get("OR_DAILY_CAP", str(DEFAULT_DAILY_CAP)))
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> Dict[str, Any]:
        if self.path.is_file():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "date": self.path.stem.split("_")[0] if "_" in self.path.stem else date.today().isoformat(),
            "cap": self.daily_cap,
            "total": 0,
            "by_part": {"agent": 0, "defense": 0, "attacker": 0, "s4": 0},
            "status_counts": {},
            "n_429": 0,
            "consecutive_429": 0,
            "stopped": False,
            "stop_reason": None,
        }

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self.load()

    def remaining(self) -> int:
        snap = self.snapshot()
        if snap.get("stopped"):
            return 0
        return max(0, int(snap.get("cap", self.daily_cap)) - int(snap.get("total", 0)))

    def _apply_status(self, d: Dict[str, Any], status_code: Optional[int]) -> None:
        if status_code is None:
            return
        sc = dict(d.get("status_counts") or {})
        key = str(status_code)
        sc[key] = int(sc.get(key, 0)) + 1
        d["status_counts"] = sc
        if status_code == 429:
            d["n_429"] = int(d.get("n_429", 0)) + 1
            d["consecutive_429"] = int(d.get("consecutive_429", 0)) + 1
            if d["consecutive_429"] >= 3:
                d["stopped"] = True
                d["stop_reason"] = "3 consecutive 429s"
        else:
            d["consecutive_429"] = 0

    def reserve(
        self,
        part: str,
        *,
        n: int = 1,
        part_cap: Optional[int] = None,
    ) -> int:
        """
        Claim ``n`` request slots *before* sending.

        Raises ``BudgetExceeded`` if the daily / part / stopped caps would be
        violated. Increments ``total`` immediately so concurrent callers cannot
        oversubscribe. Pair with ``commit`` after the send to adjust to the
        actual count and record status codes.
        """
        if n < 0:
            raise ValueError("reserve n must be >= 0")
        if n == 0:
            return 0
        with self._lock:
            d = self.load()
            if d.get("stopped"):
                raise BudgetExceeded(f"stopped: {d.get('stop_reason')}")
            if int(d.get("total", 0)) + n > int(d.get("cap", self.daily_cap)):
                raise BudgetExceeded("daily cap reached")
            by = dict(d.get("by_part") or {})
            cur = int(by.get(part, 0))
            if part_cap is not None and cur + n > part_cap:
                raise BudgetExceeded(f"part cap {part}={part_cap} reached")
            d["total"] = int(d.get("total", 0)) + n
            by[part] = cur + n
            d["by_part"] = by
            self.save(d)
            return n

    def commit(
        self,
        part: str,
        *,
        reserved: int = 0,
        actual: int = 0,
        status_code: Optional[int] = None,
        n_429: int = 0,
    ) -> Dict[str, Any]:
        """
        Finalize a reservation after the request(s) were sent.

        Adjusts ``total`` / ``by_part`` from ``reserved`` to ``actual`` and
        records status / 429 counters. Never raises ``BudgetExceeded`` — every
        request that went out must be persisted, even past the cap.
        """
        if reserved < 0 or actual < 0:
            raise ValueError("reserved/actual must be >= 0")
        with self._lock:
            d = self.load()
            delta = int(actual) - int(reserved)
            if delta != 0:
                d["total"] = max(0, int(d.get("total", 0)) + delta)
                by = dict(d.get("by_part") or {})
                by[part] = max(0, int(by.get(part, 0)) + delta)
                d["by_part"] = by
            # Cap overrun after the fact: mark stopped but keep the counts.
            if int(d.get("total", 0)) > int(d.get("cap", self.daily_cap)):
                d["stopped"] = True
                if not d.get("stop_reason"):
                    d["stop_reason"] = "daily cap reached"
            if status_code is not None:
                self._apply_status(d, status_code)
            for _ in range(max(0, int(n_429))):
                self._apply_status(d, 429)
            self.save(d)
            return d

    def record(
        self,
        part: str,
        *,
        n: int = 1,
        part_cap: Optional[int] = None,
        status_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Increment counters; raise BudgetExceeded if caps would be violated.

        For a send that might exceed the remaining budget after the HTTP call,
        use ``reserve`` then ``commit`` instead so already-sent requests are
        never dropped when the cap check fails.
        """
        if n > 0:
            self.reserve(part, n=n, part_cap=part_cap)
            return self.commit(
                part,
                reserved=n,
                actual=n,
                status_code=status_code,
            )
        # Status-only update (e.g. n=0, status_code=429) — never raise for
        # already-sent traffic; still refuse when the ledger is hard-stopped
        # and no new count is being applied? Prefer always persisting status
        # for sent requests. Use commit path.
        return self.commit(
            part,
            reserved=0,
            actual=0,
            status_code=status_code,
        )


def install_openai_hooks(
    openai_client: Any,
    part: str,
    ledger: Optional[RequestLedger] = None,
    *,
    part_cap: Optional[int] = None,
) -> RequestLedger:
    """
    Attach counting hooks to an ``openai.OpenAI`` client's httpx transport.

    Live research only — never call from CI mocks. Reserves one slot on
    request (raises BudgetExceeded before send if empty) and commits the
    status code on response.
    """
    led = ledger or RequestLedger()
    hx = openai_client._client

    def on_request(request):  # noqa: ARG001
        led.reserve(part, n=1, part_cap=part_cap)

    def on_response(response):
        led.commit(part, reserved=1, actual=1, status_code=int(response.status_code))

    hx.event_hooks.setdefault("request", []).append(on_request)
    hx.event_hooks.setdefault("response", []).append(on_response)
    return led
