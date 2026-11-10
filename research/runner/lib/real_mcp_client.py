"""Minimal MCP stdio client (newline-delimited JSON-RPC 2.0, protocol 2025-06-18). Stdlib only.
Used by real_mcp_driver.py (host side) and real_mcp_gateway.py (gateway -> backends)."""
import json, os, queue, subprocess, threading, time, itertools

PROTOCOL = "2025-06-18"

class MCPError(Exception):
    def __init__(self, err):
        super().__init__(err.get("message", str(err))); self.code = err.get("code"); self.err = err

class StdioMCP:
    def __init__(self, name, cmd, env=None, cwd=None, stderr_path=None, on_notify=None):
        self.name = name; self.cmd = list(cmd)
        self._ids = itertools.count(1); self._wait = {}; self._lock = threading.Lock()
        self.notifications = []; self.on_notify = on_notify
        self._err = open(stderr_path, "ab") if stderr_path else subprocess.DEVNULL
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                  env=env, cwd=cwd, bufsize=0, start_new_session=True)
        self._t = threading.Thread(target=self._reader, daemon=True); self._t.start()
        self.server_info = None; self.capabilities = None; self.raw_tools_results = []

    def _reader(self):
        for line in iter(self.p.stdout.readline, b""):
            line = line.strip()
            if not line: continue
            try: msg = json.loads(line.decode("utf-8"))
            except Exception: continue
            if "id" in msg and ("result" in msg or "error" in msg):
                with self._lock: q = self._wait.get(msg["id"])
                if q: q.put(msg)
            elif "method" in msg and "id" in msg:   # server -> client request (roots/list, ping): answer minimal
                reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {"roots": []} if msg["method"] == "roots/list" else {}}
                self._send(reply)
            elif "method" in msg:
                self.notifications.append((time.time(), msg))
                if self.on_notify: self.on_notify(self, msg)
        with self._lock:
            for q in self._wait.values(): q.put({"error": {"code": -32099, "message": "server closed"}})

    def _send(self, msg):
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock: self.p.stdin.write(data); self.p.stdin.flush()

    def request(self, method, params=None, timeout=60):
        rid = next(self._ids); q = queue.Queue()
        with self._lock: self._wait[rid] = q
        msg = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None: msg["params"] = params
        self._send(msg)
        try: resp = q.get(timeout=timeout)
        finally:
            with self._lock: self._wait.pop(rid, None)
        if "error" in resp: raise MCPError(resp["error"])
        return resp.get("result")

    def notify(self, method, params=None):
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None: msg["params"] = params
        self._send(msg)

    def initialize(self, client_name="oberih-real-mcp-host", timeout=60):
        r = self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                        "clientInfo": {"name": client_name, "version": "0.1"}}, timeout=timeout)
        self.server_info = r.get("serverInfo"); self.capabilities = r.get("capabilities")
        self.notify("notifications/initialized"); return r

    def list_tools(self):
        r = self.request("tools/list", {}); self.raw_tools_results.append(r); return r

    def call_tool(self, name, arguments, timeout=60):
        return self.request("tools/call", {"name": name, "arguments": arguments or {}}, timeout=timeout)

    def close(self):
        try: self.p.stdin.close()
        except Exception: pass
        try: self.p.wait(timeout=5)
        except Exception:
            try: self.p.terminate(); self.p.wait(timeout=3)
            except Exception:
                try: self.p.kill()
                except Exception: pass
        if self._err not in (None, subprocess.DEVNULL):
            try: self._err.close()
            except Exception: pass

def result_text(result):
    if not isinstance(result, dict): return str(result)
    parts = []
    for b in result.get("content") or []:
        parts.append(str(b.get("text", "")) if isinstance(b, dict) and b.get("type") == "text" else json.dumps(b, ensure_ascii=False))
    return "\n".join(parts)
