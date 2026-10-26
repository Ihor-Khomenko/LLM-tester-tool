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
import select
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def spawn_backend(command: Sequence[str]) -> subprocess.Popen:
    """Spawn backend MCP server; prefer mcp StdioServerParameters env hygiene."""
    env = None
    if _mcp_available():
        try:
            from mcp.client.stdio import get_default_environment

            env = get_default_environment()
        except Exception:
            env = None
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
        # Cumulative guard overhead (ms) for tests / metrics.
        self.overhead_ms: float = 0.0

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

    def _handle_host_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a response to send to the host instead of forwarding, or None."""
        method = msg.get("method")
        if method != "tools/call":
            return None
        params = msg.get("params") or {}
        name = str(params.get("name", ""))
        arguments = dict(params.get("arguments") or {})
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
            reasons = "; ".join(decision.reasons) or "blocked by MCP-Sentinel"
            return _error_response(
                msg.get("id"),
                -32001,
                f"MCP-Sentinel blocked tool call '{name}': {reasons} "
                f"(risk={decision.risk_score:.2f}, stage={decision.stage})",
            )
        # Allow / sanitize path: remember id for T1 on response.
        if "id" in msg:
            self._pending_calls[msg["id"]] = call
        return None

    def _handle_backend_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Possibly mutate backend → host messages (T0 / T1)."""
        result = msg.get("result")
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
                )
                if reg.decision == Decision.BLOCK:
                    result = dict(result)
                    result["_sentinel"] = {
                        "decision": reg.decision.value,
                        "risk": reg.risk_score,
                        "reasons": reg.reasons,
                        "action": "block",
                    }
                    msg = dict(msg)
                    msg["result"] = result

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
                    f"MCP-Sentinel blocked tool output from '{call.name}': "
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
                    self._proc.stdin.write(line)
                    self._proc.stdin.flush()
                    continue
                replacement = self._handle_host_message(msg)
                if replacement is not None:
                    sys.stdout.buffer.write(_encode(replacement))
                    sys.stdout.buffer.flush()
                    continue
                self._proc.stdin.write(_encode(msg))
                self._proc.stdin.flush()

        def backend_to_host() -> None:
            assert self._proc and self._proc.stdout
            while not stop.is_set():
                line = self._proc.stdout.readline()
                if not line:
                    stop.set()
                    break
                msg = _parse_line(line)
                if msg is None:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                    continue
                out = self._handle_backend_message(msg)
                sys.stdout.buffer.write(_encode(out))
                sys.stdout.buffer.flush()

        t1 = threading.Thread(target=host_to_backend, name="host→backend", daemon=True)
        t2 = threading.Thread(target=backend_to_host, name="backend→host", daemon=True)
        t1.start()
        t2.start()
        t1.join()
        stop.set()
        t2.join(timeout=2.0)
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.guard.logger.close()
        self.audit.close()
        return self._proc.returncode or 0

    def run_http(self, host: str = "127.0.0.1", port: int = 8765) -> int:
        """
        Optional HTTP JSON-RPC front-end (one POST = one MCP message).

        Spawns the stdio backend and relays each HTTP POST body through the
        same T0/T1/T2 hooks. Suitable for clients that speak MCP-over-HTTP
        JSON-RPC (streamable-HTTP compatible request/response).
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
                        line = proxy._proc.stdout.readline()
                        backend_msg = _parse_line(line) or {}
                        out = proxy._handle_backend_message(backend_msg)
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
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
            self.guard.logger.close()
            self.audit.close()
        return 0


def _extract_text_content(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        if "text" in result:
            return str(result["text"])
        return json.dumps(result, ensure_ascii=False)
    return str(result)


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


def main(argv: Optional[List[str]] = None) -> int:
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

    llm = None
    if cfg.enable_s4 and cfg.s4_provider:
        from llm_tester.agent.provider_adapter import build_llm_chat

        llm = build_llm_chat(cfg.s4_provider, cfg.s4_model)

    guard = build_guard_from_proxy_config(
        cfg, log_path=Path(args.log) if args.log else None, llm_provider=llm
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
