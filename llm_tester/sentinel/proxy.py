"""
MCP stdio proxy for MCP-Sentinel.

Architecture (scoped MVP)
-------------------------
The proxy is a **JSON-RPC line relay** between an MCP host (stdin/stdout) and a
backend MCP server subprocess. It intercepts:

* ``tools/list`` responses → T0 registration checks
* ``tools/call`` requests → T2 call checks (may short-circuit with an error)
* ``tools/call`` responses → T1 output tainting / optional sanitize

It does **not** re-implement the full MCP SDK server surface. The optional
``mcp`` extra is used when available to spawn the backend via
``StdioServerParameters`` helpers; otherwise a plain ``subprocess.Popen`` is used.

Usage::

    python -m llm_tester.sentinel.proxy --policy policy.yaml -- \\
        npx -y @modelcontextprotocol/server-filesystem /tmp

Or::

    llmtest sentinel-proxy -- echo '{"jsonrpc":"2.0"}'   # demo only
"""

from __future__ import annotations

import argparse
import json
import select
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llm_tester.sentinel.guard import SentinelGuard, build_sentinel
from llm_tester.sentinel.types import ToolCall, ToolSpec


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


class SentinelProxy:
    """Bidirectional stdio JSON-RPC proxy with sentinel hooks."""

    def __init__(
        self,
        backend_cmd: Sequence[str],
        guard: Optional[SentinelGuard] = None,
        *,
        user_task: str = "",
    ) -> None:
        self.backend_cmd = list(backend_cmd)
        self.guard = guard or build_sentinel()
        if user_task:
            self.guard.set_user_task(user_task)
        self._pending_calls: Dict[Any, ToolCall] = {}
        self._proc: Optional[subprocess.Popen] = None

    def _handle_host_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a response to send to the host instead of forwarding, or None."""
        method = msg.get("method")
        if method != "tools/call":
            return None
        params = msg.get("params") or {}
        name = str(params.get("name", ""))
        arguments = dict(params.get("arguments") or {})
        call = ToolCall(name=name, arguments=arguments)
        decision = self.guard.check_tool_call(call)
        if decision.decision.value in ("block", "ask_user"):
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
        # tools/list result
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
                reg = self.guard.check_tool_registration(specs)
                if reg.decision.value == "block":
                    # Still return tools but log; hard-block of list is rare —
                    # attach sentinel warning in a non-standard field for hosts
                    # that understand it; protocol-compliant hosts ignore it.
                    result = dict(result)
                    result["_sentinel"] = {
                        "decision": reg.decision.value,
                        "risk": reg.risk_score,
                        "reasons": reg.reasons,
                    }
                    msg = dict(msg)
                    msg["result"] = result

        # tools/call result → T1
        req_id = msg.get("id")
        if req_id in self._pending_calls and "result" in msg:
            call = self._pending_calls.pop(req_id)
            content = _extract_text_content(msg.get("result"))
            t1 = self.guard.on_tool_output(call.name, content)
            if (
                t1.decision.value == "sanitize"
                and t1.sanitized_content is not None
                and isinstance(msg.get("result"), dict)
            ):
                msg = dict(msg)
                msg["result"] = _replace_text_content(
                    msg["result"], t1.sanitized_content
                )
            elif t1.decision.value == "block":
                return _error_response(
                    req_id,
                    -32002,
                    f"MCP-Sentinel blocked tool output from '{call.name}': "
                    + "; ".join(t1.reasons),
                )
        return msg

    def run(self) -> int:
        self._proc = spawn_backend(self.backend_cmd)
        assert self._proc.stdin and self._proc.stdout

        def pump_stderr() -> None:
            # stderr already inherited; nothing to pump unless we redirect later.
            return

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
        return self._proc.returncode or 0


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llmtest-sentinel-proxy",
        description="MCP-Sentinel stdio JSON-RPC proxy (wrap a real MCP server).",
    )
    p.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Path to sentinel YAML policy",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL decision log path",
    )
    p.add_argument(
        "--user-task",
        default="",
        help="Trusted user task for provenance (or set via SDK later)",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Backend MCP server command (prefix with --)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("Backend command required, e.g. -- python my_mcp_server.py")
    guard = build_sentinel(policy_path=args.policy, log_path=args.log)
    proxy = SentinelProxy(cmd, guard=guard, user_task=args.user_task)
    return proxy.run()


if __name__ == "__main__":
    # Avoid unused import warning for select on some platforms.
    _ = select
    raise SystemExit(main())
