"""
Adapter: mcp-security-research ``scenarios.export`` schema ``1.0`` → TraceStep.

Native export shape (see convert_mcpsr.py from the pilot)::

    {"schema_version": "1.0", "id": "...", "attack_class": "...",
     "trace": [{"event": "user_message"|"tools_listed"|"tools_mutated"|
                         "tool_result"|"tool_call", ...}, ...]}

Checkpoint mapping used by ``llmtest import-mcp-sim`` / ``annotate_checkpoints``:

* **T0** — tool schemas at registration (``tools_listed`` / ``tools``, plus
  ``tools_mutated`` / ``tools_after`` for rug-pull)
* **T1** — untrusted tool outputs (``tool_result``); the first untrusted /
  payload-bearing result is the **injection source**
* **T2** — planned tool calls (``tool_call`` + arguments; prefer ``attack_step``)
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
    "malicious_write": "direct_harm",
    "guard_prefix_bypass": "other",
    "judge_evasion": "other",
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

    result_events = [e for e in trace if e.get("event") == "tool_result"]
    outs = []
    for e in result_events:
        item = {
            "source": str(e.get("tool_name") or e.get("name") or "tool"),
            "content": str(e.get("content") or e.get("text") or ""),
        }
        if e.get("payload_file"):
            item["payload_file"] = e.get("payload_file")
        if e.get("untrusted") is not None:
            item["untrusted"] = bool(e.get("untrusted"))
        outs.append(item)

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
    labels = data.get("labels") if isinstance(data.get("labels"), dict) else {}
    # Prefer explicit label / labels.guard_should_block → attack vs benign.
    if data.get("label"):
        label = str(data.get("label"))
    elif attack_type == "benign":
        label = "benign"
    elif labels.get("guard_should_block") is False and attack_class in (
        "benign",
        "judge_evasion",
    ):
        # Catalog "benign framing" attacks still label as attack for eval.
        label = "attack"
    else:
        label = "attack"
    label = "benign" if str(label).lower() in ("benign", "safe", "clean") else "attack"

    # Injection source: first untrusted / payload-bearing tool_result.
    injection_source: Optional[Dict[str, Any]] = None
    for e in result_events:
        content = str(e.get("content") or e.get("text") or "")
        untrusted = bool(e.get("untrusted")) or bool(e.get("payload_file"))
        if untrusted or content:
            injection_source = {
                "tool": str(e.get("tool_name") or e.get("name") or "tool"),
                "content_preview": content[:240],
                "untrusted": bool(e.get("untrusted", True)),
                "payload_file": e.get("payload_file"),
                "checkpoint": "T1",
            }
            if untrusted or attack_type == "indirect_injection":
                break

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
            "guard_should_block": labels.get("guard_should_block"),
        },
    }
    if injection_source is not None:
        rec["meta"]["injection_source"] = injection_source
    if mutated:
        mut_tools = mutated[0].get("tools") or []
        rec["tools_after"] = [
            {"name": t.get("name"), "description": t.get("description", "")}
            for t in mut_tools
            if isinstance(t, dict) and t.get("name")
        ]
    elif isinstance(data.get("tools_after"), list) and data.get("tools_after"):
        rec["tools_after"] = [
            {
                "name": t.get("name"),
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", t.get("input_schema", {})),
            }
            for t in data["tools_after"]
            if isinstance(t, dict) and t.get("name")
        ]
    if call_ev:
        rec["call"] = {
            "name": str(call_ev.get("tool_name") or call_ev.get("name") or ""),
            "arguments": dict(call_ev.get("arguments") or call_ev.get("args") or {}),
        }
        rec["meta"]["call_arguments"] = dict(rec["call"]["arguments"])
        rec["meta"]["call_attack_step"] = bool(call_ev.get("attack_step"))
    return annotate_checkpoints(rec, source_record=data)


def annotate_checkpoints(
    case: Dict[str, Any],
    *,
    source_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach ``meta.checkpoints`` describing which Sentinel hooks the case hits.

    Idempotent: safe to call on already-converted v1 cases.
    """
    out = dict(case)
    meta = dict(out.get("meta") or {})
    tools = out.get("tools") or []
    tools_after = out.get("tools_after") or []
    outputs = out.get("tool_outputs") or []
    call = out.get("call")

    t0_events: List[str] = []
    if tools:
        t0_events.append("tools_listed→check_tool_registration")
    if tools_after:
        t0_events.append("tools_mutated→check_tool_registration(tools_after)")

    t1_events: List[str] = []
    if outputs:
        src = (meta.get("injection_source") or {}).get("tool") or "tool"
        t1_events.append(f"tool_result[{src}]→on_tool_output")

    t2_events: List[str] = []
    if isinstance(call, dict) and call.get("name"):
        args = call.get("arguments") or {}
        t2_events.append(
            f"tool_call[{call.get('name')}](args={sorted(args.keys())})→check_tool_call"
        )

    # Fall back to source trace events when v1 fields are sparse.
    if source_record and isinstance(source_record.get("trace"), list):
        events = {str(e.get("event")) for e in source_record["trace"] if isinstance(e, dict)}
        if "tools_listed" in events and not t0_events:
            t0_events.append("tools_listed→check_tool_registration")
        if "tools_mutated" in events and "tools_mutated→check_tool_registration(tools_after)" not in t0_events:
            t0_events.append("tools_mutated→check_tool_registration(tools_after)")
        if "tool_result" in events and not t1_events:
            t1_events.append("tool_result→on_tool_output")
        if "tool_call" in events and not t2_events:
            t2_events.append("tool_call→check_tool_call")

    meta["checkpoints"] = {
        "T0": t0_events,
        "T1": t1_events,
        "T2": t2_events,
        "exercised": [k for k, v in (("T0", t0_events), ("T1", t1_events), ("T2", t2_events)) if v],
    }
    out["meta"] = meta
    out["schema_version"] = out.get("schema_version") or TARGET_SCHEMA
    return out
