"""Strip secrets / API keys from mappings before writing artefacts."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

# Keys (case-insensitive substrings) that must never appear in artefacts.
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "api-key",
    "secret",
    "token",
    "password",
    "passwd",
    "authorization",
    "auth_header",
    "bearer",
    "credential",
    "private_key",
    "access_key",
)

_SECRET_ENV_RE = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|authorization|bearer|credential)"
)

_REDACTED = "***REDACTED***"


def _key_is_secret(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def scrub_secrets(value: Any) -> Any:
    """Recursively redact secret-looking keys from nested structures."""
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list):
        return [scrub_secrets(v) for v in value]
    if isinstance(value, tuple):
        return tuple(scrub_secrets(v) for v in value)
    return value


def redact_mapping(
    data: Mapping[str, Any],
    *,
    extra_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return a copy of ``data`` with secret fields redacted."""
    extra = {k.lower() for k in (extra_keys or ())}
    out: Dict[str, Any] = {}
    for key, val in data.items():
        if _key_is_secret(str(key)) or str(key).lower() in extra:
            out[str(key)] = _REDACTED
        elif isinstance(val, Mapping):
            out[str(key)] = redact_mapping(val, extra_keys=extra_keys)
        elif isinstance(val, list):
            out[str(key)] = [scrub_secrets(v) for v in val]
        elif isinstance(val, str) and _looks_like_secret_string(str(key), val):
            out[str(key)] = _REDACTED
        else:
            out[str(key)] = val
    return out


def _looks_like_secret_string(key: str, value: str) -> bool:
    if _SECRET_ENV_RE.search(key) and value and value not in ("ollama", "none", ""):
        # Keep placeholder / non-secret defaults; redact anything else.
        if value.lower() in {"ollama", "dummy", "test", "none", "n/a"}:
            return False
        return True
    return False


def argv_without_secrets(argv: Iterable[str]) -> list:
    """Copy CLI argv, redacting values that follow secret-looking flags."""
    items = list(argv)
    out: list = []
    skip_next = False
    for i, arg in enumerate(items):
        if skip_next:
            out.append(_REDACTED)
            skip_next = False
            continue
        if arg.startswith("--") and _SECRET_ENV_RE.search(arg.lstrip("-")):
            if "=" in arg:
                flag, _, _val = arg.partition("=")
                out.append(f"{flag}={_REDACTED}")
            else:
                out.append(arg)
                skip_next = True
            continue
        out.append(arg)
    return out


def strip_env_secrets(environ: Mapping[str, str]) -> Dict[str, str]:
    """Return env vars safe to log (presence flags only for secrets)."""
    safe: Dict[str, str] = {}
    for key, val in environ.items():
        if _SECRET_ENV_RE.search(key):
            safe[key] = "set" if val else "unset"
        else:
            safe[key] = val
    return safe
