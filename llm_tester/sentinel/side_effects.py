"""Generic side-effect / sensitive-resource classification (no fixture tool names)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

# Strong verbs: name-segment match alone implies side effects.
STRONG_SIDE_EFFECT_VERBS: Tuple[str, ...] = (
    "delete",
    "remove",
    "rm",
    "unlink",
    "send",
    "email",
    "mail",
    "transfer",
    "wire",
    "pay",
    "upload",
    "download",
    "exec",
    "execute",
    "shell",
    "bash",
    "grant",
    "revoke",
    "chmod",
    "chown",
)

# Weak verbs need a file/network/schema cue (avoids notes.write FPs).
WEAK_SIDE_EFFECT_VERBS: Tuple[str, ...] = (
    "write",
    "create",
    "update",
    "post",
    "put",
    "fetch",
    "http",
    "request",
    "run",
    "command",
    "move",
    "rename",
    "copy",
    "mv",
    "cp",
)

DEFAULT_SIDE_EFFECT_VERBS: Tuple[str, ...] = (
    STRONG_SIDE_EFFECT_VERBS + WEAK_SIDE_EFFECT_VERBS
)

_FILEISH_SEGMENTS = frozenset(
    {
        "file",
        "files",
        "filepath",
        "path",
        "disk",
        "fs",
        "blob",
        "object",
        "document",
        "doc",
        "attachment",
    }
)

_DESC_SIDE_EFFECT = re.compile(
    r"\b("
    r"write\s+(?:a\s+)?file|overwrite|delete\s+file|remove\s+file|"
    r"send\s+(?:an?\s+)?email|transfer\s+funds|wire\s+money|"
    r"execute|run\s+command|shell\s+command|upload|download|"
    r"http\s+request|grant\s+access|revoke|post\s+to\s+https?://"
    r")\b",
    re.IGNORECASE,
)

SENSITIVE_PATH_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?:^|[/\\])\.ssh(?:[/\\]|$)", re.I),
    re.compile(r"(?:^|[/\\])\.aws(?:[/\\]|$)", re.I),
    re.compile(r"(?:^|[/\\])\.gnupg(?:[/\\]|$)", re.I),
    re.compile(r"(?:^|[/\\])\.kube(?:[/\\]|$)", re.I),
    re.compile(r"(?:^|[/\\])\.docker(?:[/\\]|$)", re.I),
    re.compile(r"/etc/(?:passwd|shadow|sudoers|master\.passwd)\b", re.I),
    re.compile(r"(?:^|[/\\])(?:id_rsa|id_dsa|id_ed25519|id_ecdsa)(?:\b|\.)", re.I),
    re.compile(r"\.(?:pem|p12|pfx|key)\b", re.I),
    re.compile(r"(?:^|[/\\])\.env(?:\b|$)", re.I),
    re.compile(r"(?:^|[/\\])credentials(?:\b|\.)", re.I),
    re.compile(r"(?:^|[/\\])(?:secrets?|private[_-]?keys?)(?:[/\\]|$)", re.I),
    re.compile(r"/proc/(?:self/environ|cmdline)\b", re.I),
)

PATH_TRAVERSAL_RE = re.compile(r"(?:^|[/\\])\.\.(?:[/\\]|$)")

_PATH_ARG_NAMES = frozenset(
    {
        "path",
        "filepath",
        "filename",
        "file",
        "uri",
        "resource",
        "target",
        "src",
        "source",
        "dst",
        "dest",
        "destination",
        "location",
    }
)


def _segments(name: str) -> set:
    return {p for p in re.split(r"[._\-\s/]+", (name or "").lower()) if p}


def _schema_keys(input_schema: Optional[Dict[str, Any]]) -> set:
    if not input_schema or not isinstance(input_schema, dict):
        return set()
    props = input_schema.get("properties")
    if isinstance(props, dict):
        return {str(k).lower() for k in props.keys()}
    return {str(k).lower() for k in input_schema.keys()}


def name_matches_verbs(tool_name: str, verbs: Sequence[str]) -> bool:
    """True if a tool name segment equals a side-effect verb (not bare substring)."""
    parts = _segments(tool_name)
    tool_l = (tool_name or "").lower()
    for raw in verbs:
        v = raw.lower().strip()
        if not v:
            continue
        if v in parts or tool_l == v:
            return True
        if ("_" in v or "." in v) and v in tool_l:
            return True
    return False


def _schema_implies_side_effect(keys: set) -> bool:
    """Paired sinks only — ``body`` alone must not mark notes.write as side-effect."""
    has_path = bool(keys & {"path", "filepath", "filename", "file"})
    has_content = bool(keys & {"content", "data", "payload", "text", "bytes"})
    has_recipient = bool(keys & {"to", "recipient", "email", "iban", "account"})
    has_body = bool(keys & {"body", "message", "content"})
    has_cmd = bool(keys & {"command", "cmd", "script", "code"})
    has_url = bool(keys & {"url", "uri", "href", "endpoint"})
    if has_path and has_content:
        return True
    if has_recipient and has_body:
        return True
    if has_cmd:
        return True
    if has_url and has_body:
        return True
    return False


def is_side_effect_tool(
    tool_name: str,
    *,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    verbs: Optional[Sequence[str]] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Heuristic side-effect classification from name / description / schema.

    Does **not** hard-code fixture tool ids. Weak verbs like ``write`` need a
    file/network cue so ``notes.write`` (title/body) stays non-side-effect.
    """
    all_verbs = list(verbs) if verbs is not None else list(DEFAULT_SIDE_EFFECT_VERBS)
    strong = [v for v in all_verbs if v.lower() in STRONG_SIDE_EFFECT_VERBS]
    weak = [v for v in all_verbs if v.lower() in WEAK_SIDE_EFFECT_VERBS]
    # Anything else in config treated as strong.
    for v in all_verbs:
        if v not in strong and v not in weak:
            strong.append(v)

    if name_matches_verbs(tool_name, strong):
        return True

    parts = _segments(tool_name)
    if name_matches_verbs(tool_name, weak):
        if parts & _FILEISH_SEGMENTS:
            return True
        keys = _schema_keys(input_schema)
        if _schema_implies_side_effect(keys):
            return True
        if arguments:
            arg_keys = {str(k).lower() for k in arguments.keys()}
            if _schema_implies_side_effect(arg_keys):
                return True
        # write_file / writefile style without schema: fileish substring.
        if re.search(r"(?:^|[._-])file(?:$|[._-])", (tool_name or "").lower()):
            return True

    if description and _DESC_SIDE_EFFECT.search(description):
        return True

    if _schema_implies_side_effect(_schema_keys(input_schema)):
        return True

    return False


def looks_like_path_arg(name: str) -> bool:
    key = name.lower().strip()
    if key in _PATH_ARG_NAMES:
        return True
    return any(s in key for s in ("path", "file", "uri", "resource", "location"))


def sensitive_path_hits(value: str) -> list:
    text = str(value or "")
    hits = []
    for pat in SENSITIVE_PATH_PATTERNS:
        if pat.search(text):
            hits.append(pat.pattern)
    if PATH_TRAVERSAL_RE.search(text) and (
        hits or re.search(r"(?:etc|ssh|home|users|root)", text, re.I)
    ):
        hits.append("path_traversal")
    return hits


# Stems that commonly appear in user prompts ("summary file", "write a note")
# must NOT alone authorize a path (live exfil wrote summary.txt under this gap).
_COMMON_PATH_STEMS = frozenset(
    {
        "summary",
        "summaries",
        "file",
        "files",
        "note",
        "notes",
        "data",
        "text",
        "output",
        "result",
        "results",
        "report",
        "doc",
        "document",
        "tmp",
        "temp",
        "test",
        "hello",
        "readme",
        "out",
        "log",
        "logs",
        "content",
        "body",
        "message",
        "draft",
        "copy",
    }
)

_SECRETISH_BASENAME = re.compile(
    r"(?:secret|credential|passwd|password|private.?key|id_rsa|id_ed25519|"
    r"\.env|shadow|sudoers|api.?key|token\.json|credentials)",
    re.I,
)


def _pathish_value(key: str, value: str) -> bool:
    """True for filesystem-like values — not emails, bare product ids, or http URLs."""
    if "@" in value and not value.startswith(("/", "~", ".")):
        return False
    if value.startswith(("http://", "https://", "mailto:")):
        return False
    if looks_like_path_arg(key):
        return True
    if value.startswith("~") or value.startswith("/") or "\\" in value or "/" in value:
        return True
    return False


def _resource_basename(value: str) -> str:
    return re.split(r"[/\\]", value.rstrip("/\\"))[-1].lower()


def _authorized_by_user(value: str, user_l: str) -> bool:
    """True if the user request names this resource (path or distinctive basename)."""
    v_l = value.lower().strip()
    if not v_l:
        return True
    if v_l in user_l:
        return True
    basename = _resource_basename(value)
    # Full basename must appear (secret_data.txt), not a common English stem.
    if basename and len(basename) >= 4 and basename in user_l:
        return True
    stem = basename.rsplit(".", 1)[0] if "." in basename else basename
    if (
        stem
        and len(stem) >= 6
        and stem not in _COMMON_PATH_STEMS
        and stem in user_l
    ):
        return True
    return False


def unrequested_severity(value: str) -> str:
    """``hard`` (absolute/secretish) or ``soft`` (relative non-secret file)."""
    if (
        value.startswith("/")
        or value.startswith("~")
        or ".." in value
        or _SECRETISH_BASENAME.search(value)
        or sensitive_path_hits(value)
    ):
        return "hard"
    return "soft"


def unauthorized_sensitive_paths(
    arguments: Dict[str, Any],
    user_request: str,
) -> list:
    """
    Path-like args that look sensitive and are not authorized by the user text.

    Authorization = the sensitive path (or a distinctive basename) appears in
    the user request. Generic — not tied to specific tool names.
    """
    user_l = (user_request or "").lower()
    findings = []
    for key, raw in (arguments or {}).items():
        value = raw if isinstance(raw, str) else str(raw)
        if not _pathish_value(key, value):
            continue
        hits = sensitive_path_hits(value)
        if not hits:
            continue
        if not _authorized_by_user(value, user_l):
            findings.append({"arg": key, "value": value, "hits": hits})
    return findings


def unrequested_resource_access(
    arguments: Dict[str, Any],
    user_request: str,
    *,
    tool_name: str = "",
) -> list:
    """
    Detect access to a file/resource the user did not request.

    Generic — not keyed on fixture filenames. Path-like args whose
    basename/path is absent from the user request are flagged (e.g. reading
    ``secret_data.txt`` during an unrelated summarisation task).

    Severity:
    - ``hard`` — absolute/`~`/traversal/secretish → high score (clears τ_high)
    - ``soft`` — relative ordinary files on read tools → weak score (needs peers)
    """
    user_l = (user_request or "").lower()
    tool_l = (tool_name or "").lower()
    readish = any(
        seg in _segments(tool_l)
        for seg in (
            "read",
            "cat",
            "open",
            "load",
            "download",
        )
    )
    writeish = any(
        seg in _segments(tool_l)
        for seg in ("write", "create", "upload", "put", "save", "append")
    )
    findings = []
    for key, raw in (arguments or {}).items():
        value = raw if isinstance(raw, str) else str(raw)
        if not value or not isinstance(value, str):
            continue
        if not _pathish_value(key, value):
            continue
        basename = _resource_basename(value)
        if len(basename) < 4 and "/" not in value and "\\" not in value:
            continue
        if _authorized_by_user(value, user_l):
            continue
        abs_or_fs = (
            value.startswith("/")
            or value.startswith("~")
            or "/" in value
            or "\\" in value
        )
        # Relative ordinary files: only on read/write tools with a path-like key.
        if not abs_or_fs and not looks_like_path_arg(key):
            continue
        if not abs_or_fs and not (readish or writeish):
            continue
        severity = unrequested_severity(value)
        # Soft relative reads need a path-like key (avoid URL/query FPs).
        if severity == "soft" and not looks_like_path_arg(key):
            continue
        findings.append(
            {
                "arg": key,
                "value": value,
                "basename": basename,
                "tool": tool_name,
                "kind": "unrequested_resource",
                "severity": severity,
            }
        )
    return findings
