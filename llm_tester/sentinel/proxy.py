"""
MCP proxy for MCP-Sentinel (stdio first; optional HTTP JSON-RPC).

Architecture
------------
Transparent JSON-RPC relay between an MCP client and a real MCP server.
Intercepts:

* ``tools/list`` responses → **T0** registration checks
* ``tools/call`` requests → **T2** call checks (may short-circuit: block)
* ``tools/call`` responses → **T1** output checks (block / sanitize / allow)

Actions: **block**, **sanitize**, **allow**. Writes JSONL audit logs with
per-hook latency. Config via YAML (``threshold`` default 0.7; S4 optional).

Usage::

    llmtest sentinel-proxy --config llm_tester/data/sentinel/proxy.default.yaml -- \\
      npx -y @modelcontextprotocol/server-filesystem /tmp

    llmtest sentinel-proxy --transport http --listen 127.0.0.1:8765 -- \\
      npx -y @modelcontextprotocol/server-filesystem /tmp
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from llm_tester.sentinel.guard import SentinelGuard, build_sentinel
from llm_tester.sentinel.policy import load_policy
from llm_tester.sentinel.proxy_config import (
    DEFAULT_THRESHOLD,
    ProxyConfig,
    apply_proxy_config_to_policy,
    load_proxy_config,
)
from llm_tester.sentinel.types import Decision, ToolCall, ToolSpec


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mcp_available() -> bool:
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        return False


# Mirror mcp.client.stdio.get_default_environment (+ locale / TZ). Always used
# so backends never inherit API keys when the ``mcp`` package is absent.
_DEFAULT_INHERITED_ENV_VARS = (
    (
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    )
    if sys.platform == "win32"
    else (
        "HOME",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "USER",
    )
)
# Locale / timezone — safe, needed by many CLIs; not in upstream mcp list.
_EXTRA_SAFE_ENV_VARS = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LANGUAGE",
    "TZ",
    "TMPDIR",
)
_SECRET_ENV_RE = re.compile(
    r"(?:_KEY|_TOKEN|SECRET|PASSWORD|PASSWD|OPENAI_|REQUESTY_|OPENROUTER_|"
    r"ANTHROPIC_|GROQ_|XAI_|HF_|AWS_|GOOGLE_)",
    re.I,
)


def get_sanitized_backend_environment(
    *,
    source: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Minimal env for MCP backend subprocesses (no API keys / tokens).

    Whitelists PATH/HOME/locale (and Windows equivalents), mirroring
    ``mcp.client.stdio.get_default_environment``, then drops any name that
    still looks like a secret. Does **not** require the ``mcp`` package.
    """
    src = source if source is not None else os.environ
    env: Dict[str, str] = {}
    for key in _DEFAULT_INHERITED_ENV_VARS + _EXTRA_SAFE_ENV_VARS:
        value = src.get(key)
        if value is None:
            continue
        if value.startswith("()"):
            continue
        if _SECRET_ENV_RE.search(key):
            continue
        env[key] = value
    return env


def spawn_backend(command: Sequence[str]) -> subprocess.Popen:
    """Spawn backend MCP server with a sanitized environment (no secret leak)."""
    env = get_sanitized_backend_environment()
    return subprocess.Popen(
        list(command),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        bufsize=0,
        env=env,
    )


def _parse_line(line: bytes) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _encode(msg: Dict[str, Any]) -> bytes:
    return (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")


def _error_response(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


class AuditLogger:
    """Append-only JSONL audit trail for proxy decisions + latency."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self._fh = None
        self.records: List[Dict[str, Any]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, **fields: Any) -> Dict[str, Any]:
        record = {"timestamp": _utc_now(), **fields}
        self.records.append(record)
        if self._fh is not None:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class SentinelProxy:
    """
    Bidirectional JSON-RPC proxy with Sentinel T0/T1/T2 hooks.

    Works with a subprocess backend (``run`` / ``run_http``) or in-process
    against injected host/backend callables (tests).
    """

    def __init__(
        self,
        backend_cmd: Optional[Sequence[str]] = None,
        guard: Optional[SentinelGuard] = None,
        *,
        user_task: str = "",
        audit: Optional[AuditLogger] = None,
        config: Optional[ProxyConfig] = None,
    ) -> None:
        self.backend_cmd = list(backend_cmd or [])
        self.config = config or ProxyConfig()
        self.guard = guard or build_sentinel()
        task = user_task or self.config.user_task
        if task:
            self.guard.set_user_task(task)
        self.audit = audit or AuditLogger(
            Path(self.config.audit_log) if self.config.audit_log else None
        )
        self._pending_calls: Dict[Any, ToolCall] = {}
        self._proc: Optional[subprocess.Popen] = None
        # Tools flagged at T0 (poison / hash mismatch / rug-pull); blocked on call
        # when ``t0_mode=enforce``.
        self._t0_flagged_tools: Set[str] = set()
        self._stdout_lock = threading.Lock()
        # HTTP: stash out-of-order responses / skip notifications without desync.
        self._http_stash: Dict[Any, Dict[str, Any]] = {}
        self._http_notifications: List[Dict[str, Any]] = []
        # Cumulative guard overhead (ms) for tests / metrics.
        self.overhead_ms: float = 0.0

    @property
    def t0_enforce(self) -> bool:
        return (self.config.t0_mode or "enforce").lower().strip() != "warn"

    def _write_host(self, data: bytes) -> None:
        """Serialize all writes to the client stdout (stdio mode)."""
        with self._stdout_lock:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    def _audit_decision(
        self,
        *,
        event: str,
        checkpoint: str,
        decision: Any,
        tool: str = "",
        overhead_ms: float = 0.0,
        **extra: Any,
    ) -> None:
        self.overhead_ms += float(overhead_ms)
        action = (
            decision.decision.value
            if hasattr(decision, "decision")
            else str(decision)
        )
        self.audit.log(
            event=event,
            checkpoint=checkpoint,
            tool=tool,
            action=action,
            risk_score=getattr(decision, "risk_score", None),
            reasons=list(getattr(decision, "reasons", None) or []),
            overhead_ms=round(float(overhead_ms), 4),
            **extra,
        )

    def _per_tool_risk(self, entry: Dict[str, Any]) -> float:
        """Reconstruct per-tool risk from ``check_tool_registration`` meta."""
        t0 = entry.get("t0") if isinstance(entry.get("t0"), dict) else {}
        s1 = entry.get("s1") if isinstance(entry.get("s1"), dict) else {}
        risk = float(s1.get("score") or 0.0)
        if t0.get("poison_phrases"):
            risk = max(risk, 0.95)
        if t0.get("action") == "hash_mismatch":
            dist = float(t0.get("semantic_distance") or 0.0)
            thr = float(
                getattr(self.guard.policy, "t0_semantic_diff_threshold", 0.35) or 0.35
            )
            if dist >= thr:
                risk = max(risk, min(1.0, 0.5 + dist))
            else:
                risk = max(risk, 0.4)
            if getattr(self.guard.policy, "t0_block_on_hash_mismatch", True):
                risk = max(risk, 0.8)
        return float(min(1.0, risk))

    def _tool_is_flagged(self, risk: float) -> bool:
        """Match T0 cascade: anything above ``tau_low`` is non-ALLOW."""
        tau_low = float(getattr(self.guard.policy, "tau_low", 0.25) or 0.25)
        return risk > tau_low

    def _handle_host_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a response to send to the host instead of forwarding, or None."""
        method = msg.get("method")
        if method != "tools/call":
            return None
        params = msg.get("params") or {}
        name = str(params.get("name", ""))
        arguments = dict(params.get("arguments") or {})

        # T0 enforce: refuse calls to tools stripped/flagged at registration.
        if self.t0_enforce and name in self._t0_flagged_tools:
            self.audit.log(
                event="t0_call_blocked",
                checkpoint="T0",
                tool=name,
                action="block",
                risk_score=None,
                reasons=["t0:flagged_tool"],
                overhead_ms=0.0,
            )
            return _error_response(
                msg.get("id"),
                -32001,
                f"Oberih blocked tool call '{name}': flagged at T0 "
                "(poisoned description / hash pin mismatch / rug-pull)",
            )

        call = ToolCall(name=name, arguments=arguments)
        t0 = time.perf_counter()
        decision = self.guard.check_tool_call(call)
        overhead = (time.perf_counter() - t0) * 1000.0
        self._audit_decision(
            event="t2_check",
            checkpoint="T2",
            decision=decision,
            tool=name,
            overhead_ms=overhead,
            arguments=arguments,
        )
        if decision.decision in (Decision.BLOCK, Decision.ASK_USER):
            reasons = "; ".join(decision.reasons) or "blocked by Oberih"
            return _error_response(
                msg.get("id"),
                -32001,
                f"Oberih blocked tool call '{name}': {reasons} "
                f"(risk={decision.risk_score:.2f}, stage={decision.stage})",
            )
        # Allow / sanitize path: remember id for T1 on response.
        if "id" in msg:
            self._pending_calls[msg["id"]] = call
        return None

    def _apply_t0_to_tools_list(
        self,
        msg: Dict[str, Any],
        *,
        tools_raw: List[Any],
        reg: Any,
    ) -> Dict[str, Any]:
        """Strip or annotate flagged tools on a tools/list result."""
        result = dict(msg.get("result") or {})
        flagged: List[str] = []
        kept: List[Any] = []
        details_by_name = {
            str(e.get("name") or ""): e
            for e in (reg.meta or {}).get("tools") or []
            if isinstance(e, dict)
        }
        for t in tools_raw:
            if not isinstance(t, dict):
                kept.append(t)
                continue
            name = str(t.get("name") or "")
            entry = details_by_name.get(name) or {"name": name, "t0": {}, "s1": {}}
            risk = self._per_tool_risk(entry)
            if self._tool_is_flagged(risk):
                flagged.append(name)
                self._t0_flagged_tools.add(name)
            else:
                kept.append(t)
                # Recovered / clean re-list: allow again if previously flagged.
                self._t0_flagged_tools.discard(name)

        result["tools"] = kept if self.t0_enforce else list(tools_raw)
        if flagged or reg.decision in (Decision.BLOCK, Decision.ASK_USER):
            result["_sentinel"] = {
                "decision": reg.decision.value,
                "risk": reg.risk_score,
                "reasons": list(reg.reasons or []),
                "action": "strip" if (self.t0_enforce and flagged) else "warn",
                "flagged_tools": flagged,
                "t0_mode": "enforce" if self.t0_enforce else "warn",
                "removed": bool(self.t0_enforce and flagged),
            }
        out = dict(msg)
        out["result"] = result
        return out

    def _handle_backend_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Possibly mutate backend → host messages (T0 / T1)."""
        result = msg.get("result")
        # tools/list (and re-list after notifications/tools/list_changed)
        if isinstance(result, dict) and "tools" in result and "id" in msg:
            tools_raw = result.get("tools") or []
            specs = []
            for t in tools_raw:
                if not isinstance(t, dict):
                    continue
                specs.append(
                    ToolSpec(
                        name=str(t.get("name", "")),
                        description=str(t.get("description", "")),
                        input_schema=t.get("inputSchema"),
                    )
                )
            if specs:
                t0 = time.perf_counter()
                reg = self.guard.check_tool_registration(specs)
                overhead = (time.perf_counter() - t0) * 1000.0
                self._audit_decision(
                    event="t0_check",
                    checkpoint="T0",
                    decision=reg,
                    tool=",".join(s.name for s in specs),
                    overhead_ms=overhead,
                    n_tools=len(specs),
                    t0_mode="enforce" if self.t0_enforce else "warn",
                )
                msg = self._apply_t0_to_tools_list(
                    msg, tools_raw=list(tools_raw), reg=reg
                )

        req_id = msg.get("id")
        if req_id in self._pending_calls and "result" in msg:
            call = self._pending_calls.pop(req_id)
            content = _extract_text_content(msg.get("result"))
            t0 = time.perf_counter()
            t1 = self.guard.on_tool_output(call.name, content)
            overhead = (time.perf_counter() - t0) * 1000.0
            self._audit_decision(
                event="t1_check",
                checkpoint="T1",
                decision=t1,
                tool=call.name,
                overhead_ms=overhead,
            )
            if (
                t1.decision == Decision.SANITIZE
                and t1.sanitized_content is not None
                and isinstance(msg.get("result"), dict)
            ):
                msg = dict(msg)
                msg["result"] = _replace_text_content(
                    msg["result"], t1.sanitized_content
                )
            elif t1.decision in (Decision.BLOCK, Decision.ASK_USER):
                return _error_response(
                    req_id,
                    -32002,
                    f"Oberih blocked tool output from '{call.name}': "
                    + "; ".join(t1.reasons),
                )
        return msg

    def exchange(
        self,
        host_msg: Dict[str, Any],
        backend_handler: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        In-process round-trip for tests: host → (optional block) → backend → T0/T1.

        ``backend_handler`` receives the forwarded request and returns a
        JSON-RPC response (as the MCP server would).
        """
        blocked = self._handle_host_message(host_msg)
        if blocked is not None:
            return blocked
        backend_resp = backend_handler(host_msg)
        return self._handle_backend_message(backend_resp)

    def _terminate_backend(self, *, wait_s: float = 0.5) -> None:
        """Close stdin + terminate backend so reader threads unblock quickly."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass

    def run(self) -> int:
        if not self.backend_cmd:
            raise RuntimeError("Backend command required for stdio proxy")
        self._proc = spawn_backend(self.backend_cmd)
        assert self._proc.stdin and self._proc.stdout

        stop = threading.Event()

        def host_to_backend() -> None:
            assert self._proc and self._proc.stdin
            while not stop.is_set():
                line = sys.stdin.buffer.readline()
                if not line:
                    stop.set()
                    break
                msg = _parse_line(line)
                if msg is None:
                    try:
                        self._proc.stdin.write(line)
                        self._proc.stdin.flush()
                    except (BrokenPipeError, ValueError):
                        stop.set()
                        break
                    continue
                replacement = self._handle_host_message(msg)
                if replacement is not None:
                    self._write_host(_encode(replacement))
                    continue
                try:
                    self._proc.stdin.write(_encode(msg))
                    self._proc.stdin.flush()
                except (BrokenPipeError, ValueError):
                    stop.set()
                    break

        def backend_to_host() -> None:
            assert self._proc and self._proc.stdout
            while not stop.is_set():
                line = self._proc.stdout.readline()
                if not line:
                    stop.set()
                    break
                msg = _parse_line(line)
                if msg is None:
                    self._write_host(line)
                    continue
                out = self._handle_backend_message(msg)
                self._write_host(_encode(out))

        t_host = threading.Thread(
            target=host_to_backend, name="host→backend", daemon=True
        )
        t_backend = threading.Thread(
            target=backend_to_host, name="backend→host", daemon=True
        )
        t_host.start()
        t_backend.start()
        try:
            t_host.join()
        finally:
            stop.set()
            # Terminate backend first so stdout readline returns immediately
            # instead of waiting up to join timeout on a live pipe.
            self._terminate_backend(wait_s=0.5)
            t_backend.join(timeout=0.2)
        self.guard.logger.close()
        self.audit.close()
        return int(self._proc.returncode or 0)

    def _is_jsonrpc_notification(self, msg: Dict[str, Any]) -> bool:
        """True for JSON-RPC notifications (method, no response id)."""
        if "method" not in msg:
            return False
        if "result" in msg or "error" in msg:
            return False
        return "id" not in msg or msg.get("id") is None

    def _read_correlated_backend_response(
        self, req_id: Any, *, max_lines: int = 64
    ) -> Dict[str, Any]:
        """
        Read backend stdout until the JSON-RPC response for ``req_id`` arrives.

        Notifications are recorded (and not treated as the response). Out-of-order
        responses for other ids are stashed for a later request.
        """
        if req_id in self._http_stash:
            return self._handle_backend_message(self._http_stash.pop(req_id))

        assert self._proc and self._proc.stdout
        for _ in range(max_lines):
            line = self._proc.stdout.readline()
            if not line:
                break
            backend_msg = _parse_line(line)
            if backend_msg is None:
                continue
            if self._is_jsonrpc_notification(backend_msg):
                self._http_notifications.append(backend_msg)
                self.audit.log(
                    event="backend_notification",
                    checkpoint="proxy",
                    tool="",
                    action="forward_skip_http",
                    risk_score=None,
                    reasons=[str(backend_msg.get("method") or "notification")],
                    overhead_ms=0.0,
                    method=backend_msg.get("method"),
                )
                continue
            mid = backend_msg.get("id")
            if mid == req_id:
                return self._handle_backend_message(backend_msg)
            if mid is not None:
                self._http_stash[mid] = backend_msg
        return _error_response(
            req_id,
            -32000,
            "Oberih proxy: timed out waiting for backend response "
            f"(id={req_id!r})",
        )

    def run_http(self, host: str = "127.0.0.1", port: int = 8765) -> int:
        """
        Optional HTTP JSON-RPC front-end (one POST = one MCP message).

        Spawns the stdio backend and relays each HTTP POST body through the
        same T0/T1/T2 hooks. Correlates backend responses by JSON-RPC id so
        interleaved notifications do not desync the session.
        """
        if not self.backend_cmd:
            raise RuntimeError("Backend command required for HTTP proxy")
        self._proc = spawn_backend(self.backend_cmd)
        assert self._proc.stdin and self._proc.stdout
        proxy = self
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                try:
                    msg = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"invalid json"}')
                    return
                with lock:
                    blocked = proxy._handle_host_message(msg)
                    if blocked is not None:
                        out = blocked
                    else:
                        assert proxy._proc and proxy._proc.stdin and proxy._proc.stdout
                        proxy._proc.stdin.write(_encode(msg))
                        proxy._proc.stdin.flush()
                        if "id" not in msg:
                            # Client notification — no correlated response.
                            out = {"jsonrpc": "2.0", "result": {"ok": True}}
                        else:
                            out = proxy._read_correlated_backend_response(msg.get("id"))
                payload = json.dumps(out, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer((host, port), Handler)
        print(
            f"sentinel-proxy HTTP listening on http://{host}:{port} "
            f"(backend={' '.join(self.backend_cmd)})",
            file=sys.stderr,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            self._terminate_backend(wait_s=0.5)
            self.guard.logger.close()
            self.audit.close()
        return 0


def _structured_content_text(structured: Any) -> str:
    if structured is None:
        return ""
    if isinstance(structured, str):
        return structured
    if isinstance(structured, dict):
        parts: List[str] = []
        for key in ("text", "content", "message", "data"):
            val = structured.get(key)
            if isinstance(val, str):
                parts.append(val)
        if parts:
            return "\n".join(parts)
        return json.dumps(structured, ensure_ascii=False)
    if isinstance(structured, list):
        return "\n".join(_structured_content_text(x) for x in structured)
    return str(structured)


def _extract_text_content(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        parts: List[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
        elif "text" in result:
            parts.append(str(result["text"]))
        # server-filesystem ≥2.x duplicates body in structuredContent — scan it.
        if "structuredContent" in result:
            sc_text = _structured_content_text(result.get("structuredContent"))
            if sc_text:
                parts.append(sc_text)
        if parts:
            return "\n".join(parts)
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def _replace_structured_content(structured: Any, text: str) -> Any:
    if isinstance(structured, str):
        return text
    if isinstance(structured, dict):
        out = dict(structured)
        replaced = False
        for key in ("text", "content", "message", "data"):
            if key in out and isinstance(out[key], str):
                out[key] = text
                replaced = True
                break
        if not replaced:
            return {"text": text}
        return out
    return {"text": text}


def _replace_text_content(result: Dict[str, Any], text: str) -> Dict[str, Any]:
    out = dict(result)
    content = out.get("content")
    if isinstance(content, list) and content:
        new_content = []
        replaced = False
        for block in content:
            if (
                not replaced
                and isinstance(block, dict)
                and block.get("type") == "text"
            ):
                new_content.append({"type": "text", "text": text})
                replaced = True
            else:
                new_content.append(block)
        if not replaced:
            new_content.append({"type": "text", "text": text})
        out["content"] = new_content
    else:
        out["content"] = [{"type": "text", "text": text}]
    if "structuredContent" in out:
        out["structuredContent"] = _replace_structured_content(
            out.get("structuredContent"), text
        )
    return out


def build_guard_from_proxy_config(
    cfg: ProxyConfig,
    *,
    log_path: Optional[Path] = None,
    llm_provider: Any = None,
) -> SentinelGuard:
    """Construct a SentinelGuard honouring proxy YAML threshold / S4 flags."""
    policy_path = Path(cfg.policy) if cfg.policy else None
    policy = load_policy(policy_path)
    apply_proxy_config_to_policy(policy, cfg)
    audit = cfg.audit_log
    decision_log = log_path or (Path(audit) if audit else None)
    return build_sentinel(
        policy=policy,
        log_path=decision_log,
        llm_provider=llm_provider,
        enable_s4=bool(cfg.enable_s4) if cfg.enable_s4 else None,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmtest-sentinel-proxy",
        description="MCP-Sentinel transparent proxy (stdio JSON-RPC; optional HTTP).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Proxy YAML (threshold, enable_s4, audit_log, transport)",
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Full Sentinel policy YAML (optional; threshold still applied)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Legacy decision JSONL path (also used as audit if --audit-log unset)",
    )
    p.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="JSONL audit log for T0/T1/T2 actions + latency",
    )
    p.add_argument(
        "--user-task",
        default="",
        help="Trusted user task for provenance",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Block threshold tau_high (default {DEFAULT_THRESHOLD})",
    )
    p.add_argument(
        "--enable-s4",
        action="store_true",
        help="Enable optional S4 judge (needs --provider)",
    )
    p.add_argument(
        "--provider",
        default=None,
        help="LLM provider for optional S4 (ollama|openrouter|…)",
    )
    p.add_argument("--model", default=None, help="S4 model id")
    p.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="stdio (default) or http JSON-RPC front-end",
    )
    p.add_argument(
        "--listen",
        default=None,
        help="host:port for --transport http (default 127.0.0.1:8765)",
    )
    p.add_argument(
        "--t0-mode",
        choices=["enforce", "warn"],
        default=None,
        help="T0 action on flagged tools: enforce (strip+block calls, default) "
        "or warn (annotate _sentinel only)",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Backend MCP server command (prefix with --)",
    )
    return p


def _parse_listen(listen: str) -> Tuple[str, int]:
    if ":" in listen:
        host, _, port_s = listen.rpartition(":")
        return host or "127.0.0.1", int(port_s)
    return "127.0.0.1", int(listen)


def main(
    argv: Optional[List[str]] = None,
    *,
    llm_provider: Any = None,
) -> int:
    """
    CLI entry. Optional ``llm_provider`` is injected by the host CLI when
    ``--enable-s4`` is set — Sentinel never imports host provider adapters.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    overrides: Dict[str, Any] = {}
    if args.policy:
        overrides["policy"] = str(args.policy)
    if args.user_task:
        overrides["user_task"] = args.user_task
    if args.threshold is not None:
        overrides["threshold"] = args.threshold
    if args.enable_s4:
        overrides["enable_s4"] = True
    if args.provider:
        overrides["s4_provider"] = args.provider
    if args.model:
        overrides["s4_model"] = args.model
    if args.transport:
        overrides["transport"] = args.transport
    if args.listen:
        overrides["listen"] = args.listen
    if args.t0_mode:
        overrides["t0_mode"] = args.t0_mode
    audit = args.audit_log or args.log
    if audit:
        overrides["audit_log"] = str(audit)

    cfg = load_proxy_config(args.config, overrides=overrides)
    if cmd:
        cfg.backend = cmd
    elif cfg.backend:
        cmd = list(cfg.backend)

    if not cmd:
        parser.error(
            "Backend command required, e.g. -- npx -y "
            "@modelcontextprotocol/server-filesystem /tmp"
        )

    if cfg.enable_s4 and llm_provider is None and cfg.s4_provider:
        print(
            "Warning: enable_s4 set but no llm_provider injected; "
            "S4 will be unavailable (pass provider via llmtest CLI).",
            file=sys.stderr,
        )

    guard = build_guard_from_proxy_config(
        cfg,
        log_path=Path(args.log) if args.log else None,
        llm_provider=llm_provider,
    )
    audit_logger = AuditLogger(Path(cfg.audit_log) if cfg.audit_log else None)
    proxy = SentinelProxy(
        cmd, guard=guard, user_task=cfg.user_task, audit=audit_logger, config=cfg
    )

    if cfg.transport == "http":
        host, port = _parse_listen(cfg.listen or "127.0.0.1:8765")
        return proxy.run_http(host=host, port=port)
    return proxy.run()


if __name__ == "__main__":
    _ = select
    raise SystemExit(main())
