"""
Adapter: mcp-security-research ``scenarios.export`` schema ``1.0`` → TraceStep.

Native export shape (see convert_mcpsr.py from the pilot)::

    {"schema_version": "1.0", "id": "...", "attack_class": "...",
     "trace": [{"event": "user_message"|"tools_listed"|"tools_mutated"|
                         "tool_result"|"tool_call", ...}, ...]}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm_tester.sentinel.types import ToolCall, ToolSpec

# Keep in sync with llm_tester.agent.traces.SCHEMA_VERSION after convert.
TARGET_SCHEMA = "mcp-sentinel-trace/v1"

MSR_SCHEMA_VERSIONS = frozenset({"1.0", "1.0.0"})

ATTACK_CLASS_MAP = {
    "indirect_prompt_injection": "indirect_injection",
    "tool_poisoning": "tool_poisoning",
    "rug_pull": "rug_pull",
    "path_traversal": "data_exfiltration",
    "benign": "benign",
    "direct_harm": "direct_harm",
    "data_exfiltration": "data_exfiltration",
}


def is_mcpsr_record(data: Dict[str, Any]) -> bool:
    """Detect mcp-security-research export records."""
    ver = str(data.get("schema_version") or data.get("schema") or "").strip()
    if ver in MSR_SCHEMA_VERSIONS:
        return True
    # Heuristic: attack_class + trace events without v1 fields.
    if "attack_class" in data and isinstance(data.get("trace"), list):
        if not data.get("attack_type") and not data.get("user_request"):
            return True
    return False


def _tool_from_export(t: Dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        name=str(t.get("name") or ""),
        description=str(t.get("description") or ""),
        input_schema=t.get("parameters") or t.get("input_schema") or t.get("inputSchema"),
    )


def convert_mcpsr_record(data: Dict[str, Any], *, default_id: str = "") -> Dict[str, Any]:
    """
    Convert one MSR schema 1.0 object into a ``mcp-sentinel-trace/v1`` dict.

    Raises ``ValueError`` if the record is missing required structure.
    """
    if not isinstance(data, dict):
        raise ValueError("mcp-security-research record must be a JSON object")
    trace = data.get("trace")
    if not isinstance(trace, list):
        raise ValueError(
            f"mcp-security-research schema 1.0 record {data.get('id')!r} "
            "requires a 'trace' array"
        )

    user = next(
        (
            str(e.get("content") or "")
            for e in trace
            if e.get("event") == "user_message"
        ),
        None,
    )
    if not user:
        user = str(
            data.get("user_prompt")
            or data.get("description")
            or data.get("user_request")
            or ""
        )

    listed = [e for e in trace if e.get("event") == "tools_listed"]
    mutated = [e for e in trace if e.get("event") == "tools_mutated"]
    tools_raw = listed[0].get("tools") if listed else data.get("tools") or []
    if not isinstance(tools_raw, list):
        tools_raw = []

    outs = [
        {
            "source": str(e.get("tool_name") or e.get("name") or "tool"),
            "content": str(e.get("content") or e.get("text") or ""),
        }
        for e in trace
        if e.get("event") == "tool_result"
    ]

    calls = [e for e in trace if e.get("event") == "tool_call"]
    attack_calls = [c for c in calls if c.get("attack_step")]
    call_ev: Optional[Dict[str, Any]] = None
    if attack_calls:
        call_ev = attack_calls[0]
    elif calls:
        # Malicious call is typically the last planned call after tool results.
        call_ev = calls[-1]

    attack_class = str(data.get("attack_class") or "other")
    attack_type = ATTACK_CLASS_MAP.get(attack_class, "other")
    label = str(data.get("label") or ("benign" if attack_type == "benign" else "attack"))

    rec: Dict[str, Any] = {
        "schema_version": TARGET_SCHEMA,
        "id": str(data.get("id") or default_id),
        "label": label,
        "attack_type": attack_type,
        "user_request": user,
        "tools": [
            {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", t.get("input_schema", {})),
            }
            for t in tools_raw
            if isinstance(t, dict) and t.get("name")
        ],
        "tool_outputs": outs,
        "meta": {
            "source": "mcp-security-research",
            "attack_class": attack_class,
            "n_calls": len(calls),
            "converted_from": str(data.get("schema_version") or "1.0"),
        },
    }
    if mutated:
        mut_tools = mutated[0].get("tools") or []
        rec["tools_after"] = [
            {"name": t.get("name"), "description": t.get("description", "")}
            for t in mut_tools
            if isinstance(t, dict) and t.get("name")
        ]
    if call_ev:
        rec["call"] = {
            "name": str(call_ev.get("tool_name") or call_ev.get("name") or ""),
            "arguments": dict(call_ev.get("arguments") or call_ev.get("args") or {}),
        }
    return rec
