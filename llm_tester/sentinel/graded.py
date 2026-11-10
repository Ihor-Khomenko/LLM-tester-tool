"""Opt-in graded action responses (Progent-style tiering) for Oberih.

When ``response_mode: graded`` is set, a cascade **block** of a local file
write that carries no canary / tainted secret is downgraded to
``Decision.QUARANTINE`` instead of a hard block. Exfil-capable, destructive,
and secret-bearing calls stay hard-blocked.

Does **not** change τ, stage weights, S3 scale, or provenance — only the
post-decision action.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Sequence, Set

from llm_tester.sentinel.side_effects import (
    STRONG_SIDE_EFFECT_VERBS,
    is_side_effect_tool,
    looks_like_path_arg,
    sensitive_path_hits,
    unrequested_severity,
)
from llm_tester.sentinel.types import Decision, GuardDecision, ToolCall

# Tools / name cues that must never be quarantined (exfil / money / shell).
_EXFIL_NAME_RE = re.compile(
    r"(?:email|mail|send|post|http|fetch|upload|transfer|wire|pay|bank|"
    r"shell|exec|bash|command|grant|revoke|delete|remove|\brm\b)",
    re.I,
)

_WRITE_NAME_RE = re.compile(
    r"(?:^|[._\-])(?:write|create|save|append|put)(?:$|[._\-])|"
    r"write_file|notes\.write|\.write$",
    re.I,
)

# Content that looks like a planted canary / secret payload.
_SECRET_CONTENT_RE = re.compile(
    r"(?:"
    r"PASSWORD\s*=|SECRET\s*=|API[_-]?KEY\s*=|TOKEN\s*=|"
    r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY|"
    r"SYSTEM_ADMIN_PASSWORD|"
    r"canary[_-]?token|CANARY|"
    r"sk-[A-Za-z0-9]{16,}|"
    r"-----BEGIN"
    r")",
    re.I,
)

DEFAULT_QUARANTINE_DIR = "/tmp/oberih-quarantine"


def is_local_file_write(call: ToolCall) -> bool:
    """True for local filesystem / note write tools (not network sinks)."""
    name = str(call.name or "")
    if _EXFIL_NAME_RE.search(name):
        return False
    if not _WRITE_NAME_RE.search(name) and not is_side_effect_tool(
        name, arguments=call.arguments
    ):
        return False
    # Prefer tools that take a path/file argument or notes.write body.
    args = call.arguments or {}
    has_path = any(
        looks_like_path_arg(str(k)) and str(v or "").strip()
        for k, v in args.items()
    )
    has_content = any(
        str(k).lower() in ("content", "body", "text", "data", "message")
        for k in args
    )
    if has_path:
        return True
    if name.lower() in ("notes.write", "write_file", "write") and has_content:
        return True
    # write_file style without schema still counts.
    return bool(_WRITE_NAME_RE.search(name))


def is_exfil_or_destructive(call: ToolCall) -> bool:
    name = str(call.name or "")
    if _EXFIL_NAME_RE.search(name):
        return True
    for verb in STRONG_SIDE_EFFECT_VERBS:
        if verb in name.lower().replace("-", "_").split("_") or verb in name.lower().split(
            "."
        ):
            if verb in (
                "send",
                "email",
                "mail",
                "transfer",
                "wire",
                "pay",
                "upload",
                "exec",
                "shell",
                "bash",
                "delete",
                "remove",
                "rm",
                "grant",
                "revoke",
            ):
                return True
    args = call.arguments or {}
    for key, val in args.items():
        kl = str(key).lower()
        if kl in ("url", "uri", "to", "email", "iban", "recipient", "command", "cmd"):
            if str(val or "").strip():
                return True
    return False


def _write_content_blob(call: ToolCall) -> str:
    args = call.arguments or {}
    parts = []
    for key in ("content", "body", "text", "data", "message", "value"):
        if key in args and args[key] is not None:
            parts.append(str(args[key]))
    return "\n".join(parts)


def _write_path(call: ToolCall) -> str:
    args = call.arguments or {}
    for key in ("path", "file", "filename", "filepath", "target"):
        if key in args and args[key]:
            return str(args[key])
    return ""


def content_has_canary_or_taint(
    call: ToolCall,
    *,
    tainted_tokens: Optional[Sequence[str]] = None,
    extra_canaries: Optional[Sequence[str]] = None,
) -> bool:
    """True when write body carries secret/canary material or high-signal taint."""
    blob = _write_content_blob(call)
    if not blob.strip():
        return False
    if _SECRET_CONTENT_RE.search(blob):
        return True
    for c in extra_canaries or ():
        if c and str(c) in blob:
            return True
    # Tainted tokens of length ≥ 6 that appear in the write body.
    blob_l = blob.lower()
    for tok in tainted_tokens or ():
        t = str(tok or "").strip().lower()
        if len(t) >= 6 and t in blob_l:
            return True
    path = _write_path(call)
    if path and (
        unrequested_severity(path) == "hard" or sensitive_path_hits(path)
    ):
        return True
    return False


def collect_tainted_tokens(store: Any) -> Set[str]:
    """Best-effort tainted token harvest from a ProvenanceStore-like object."""
    out: Set[str] = set()
    if store is None:
        return out
    # Prefer explicit API if present.
    for attr in ("tainted_tokens", "untrusted_tokens"):
        vals = getattr(store, attr, None)
        if vals:
            out.update(str(x) for x in vals)
    sources = getattr(store, "sources", None) or []
    for src in sources:
        content = getattr(src, "content", None)
        if content is None and isinstance(src, dict):
            content = src.get("content")
        if not content:
            continue
        # Pull password-like lines and dense tokens from untrusted outputs.
        for m in _SECRET_CONTENT_RE.finditer(str(content)):
            out.add(m.group(0))
        for tok in re.findall(r"[A-Za-z0-9_\-]{8,}", str(content)):
            if tok.lower() not in {"important", "ignore", "previous", "instructions"}:
                out.add(tok)
    return out


def should_quarantine(
    call: ToolCall,
    *,
    prior_decision: Decision,
    response_mode: str = "block",
    tainted_tokens: Optional[Sequence[str]] = None,
    extra_canaries: Optional[Sequence[str]] = None,
) -> bool:
    """
    True when a hard-block may be softened to quarantine under graded mode.

    Requires: graded mode + prior BLOCK/ASK_USER + local file write + no
    exfil/destructive + no canary/taint in content/path.
    """
    mode = (response_mode or "block").lower().strip()
    if mode not in ("graded", "quarantine", "tiered"):
        return False
    if prior_decision not in (Decision.BLOCK, Decision.ASK_USER):
        return False
    if is_exfil_or_destructive(call):
        return False
    if not is_local_file_write(call):
        return False
    if content_has_canary_or_taint(
        call, tainted_tokens=tainted_tokens, extra_canaries=extra_canaries
    ):
        return False
    return True


def quarantine_path_for(
    original_path: str,
    *,
    quarantine_dir: str = DEFAULT_QUARANTINE_DIR,
) -> str:
    base = os.path.basename(str(original_path or "out.txt").replace("\\", "/")) or "out.txt"
    root = quarantine_dir.rstrip("/\\") or DEFAULT_QUARANTINE_DIR
    return f"{root}/{base}"


def apply_graded_decision(
    result: GuardDecision,
    call: ToolCall,
    *,
    response_mode: str,
    quarantine_dir: str = DEFAULT_QUARANTINE_DIR,
    tainted_tokens: Optional[Sequence[str]] = None,
    extra_canaries: Optional[Sequence[str]] = None,
) -> GuardDecision:
    """Possibly rewrite a BLOCK into QUARANTINE; otherwise return unchanged."""
    if not should_quarantine(
        call,
        prior_decision=result.decision,
        response_mode=response_mode,
        tainted_tokens=tainted_tokens,
        extra_canaries=extra_canaries,
    ):
        return result
    original = _write_path(call) or "(no-path)"
    qpath = quarantine_path_for(original, quarantine_dir=quarantine_dir)
    meta = dict(result.meta or {})
    meta.update(
        {
            "graded_action": "quarantine",
            "response_mode": "graded",
            "original_path": original,
            "quarantine_path": qpath,
            "prior_decision": result.decision.value,
        }
    )
    reasons = list(result.reasons or []) + [
        "graded:quarantine_local_write",
        f"graded:redirect:{qpath}",
    ]
    return GuardDecision(
        decision=Decision.QUARANTINE,
        risk_score=result.risk_score,
        stage=result.stage,
        checkpoint=result.checkpoint,
        reasons=reasons,
        latency_ms=result.latency_ms,
        stage_scores=list(result.stage_scores or []),
        sanitized_content=result.sanitized_content,
        escalated=result.escalated,
        meta=meta,
    )


def rewrite_call_for_quarantine(
    call: ToolCall,
    decision: GuardDecision,
) -> ToolCall:
    """Return a copy of ``call`` with path args redirected to quarantine."""
    qpath = (decision.meta or {}).get("quarantine_path")
    if not qpath:
        return call
    args = dict(call.arguments or {})
    for key in ("path", "file", "filename", "filepath", "target"):
        if key in args and args[key]:
            args[key] = qpath
            break
    else:
        args["path"] = qpath
    return ToolCall(name=call.name, arguments=args)
