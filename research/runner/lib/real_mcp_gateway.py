"""Local MCP gateway (aggregator): one stdio MCP server in front of N stdio backends (merged tools/list, tools/call routed
by tool name, notifications/tools/list_changed forwarded). Used only for the `oberih_gw` arm so ONE Oberih proxy (one guard,
one provenance store) sees all servers. Backends from $GW_BACKENDS (JSON: [{"name":..,"cmd":[..],"cwd":..}]). Stdlib only."""
import json, os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_mcp_client import StdioMCP, MCPError, PROTOCOL

out_lock = threading.Lock()
def emit(msg):
    with out_lock:
        sys.stdout.buffer.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")); sys.stdout.buffer.flush()

def on_notify(_b, msg):
    if msg.get("method") == "notifications/tools/list_changed": emit({"jsonrpc": "2.0", "method": msg["method"]})

backs = []; route = {}
for b in json.loads(os.environ["GW_BACKENDS"]):
    c = StdioMCP(b["name"], b["cmd"], env=os.environ.copy(), cwd=b.get("cwd"), stderr_path=os.environ.get("GW_STDERR"), on_notify=on_notify)
    c.initialize(client_name="oberih-local-gateway"); backs.append(c)

def handle(msg):
    m = msg.get("method"); rid = msg.get("id")
    try:
        if m == "initialize":
            res = {"protocolVersion": PROTOCOL, "capabilities": {"tools": {"listChanged": True}}, "serverInfo": {"name": "local-gateway", "version": "0.1"}}
        elif m == "tools/list":
            tools = []
            for c in backs:
                for t in (c.list_tools() or {}).get("tools", []):
                    route[t["name"]] = c; tools.append(t)
            res = {"tools": tools}
        elif m == "tools/call":
            name = (msg.get("params") or {}).get("name")
            c = route.get(name)
            if c is None: raise MCPError({"code": -32602, "message": f"unknown tool {name}"})
            res = c.call_tool(name, (msg.get("params") or {}).get("arguments") or {})
        elif m == "ping":
            res = {}
        elif rid is None:
            return
        else:
            raise MCPError({"code": -32601, "message": f"method not found: {m}"})
        emit({"jsonrpc": "2.0", "id": rid, "result": res})
    except MCPError as e:
        if rid is not None: emit({"jsonrpc": "2.0", "id": rid, "error": e.err})

for line in iter(sys.stdin.buffer.readline, b""):
    line = line.strip()
    if not line: continue
    try: msg = json.loads(line.decode("utf-8"))
    except Exception: continue
    handle(msg)
for c in backs: c.close()
