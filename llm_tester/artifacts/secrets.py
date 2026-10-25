"""Strip secrets / API keys from mappings before writing artefacts.

Masks real credentials (API keys, auth headers, key-shaped values). Does
**not** redact metric / provenance fields whose names merely contain the
substring ``token`` (e.g. ``tokens``, ``tainted_tokens``, ``token_pi``,
``token_sim``, ``max_tokens``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional

# Exact / suffix-style secret *key* names (after lowercasing + '-' → '_').
_SECRET_KEY_EXACT = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "access_key",
        "secret",
        "secret_key",
        "password",
        "passwd",
        "authorization",
        "auth_header",
        "auth",
        "bearer",
        "credential",
        "credentials",
        "private_key",
        "token",  # exact "token" only — not "tokens" / "token_pi"
        "access_token",
        "id_token",
        "refresh_token",
        "auth_token",
        "session_token",
        "openai_api_key",
        "openrouter_api_key",
        "xai_api_key",
        "groq_api_key",
        "anthropic_api_key",
    }
)

_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_access_key",
    "_secret",
    "_password",
    "_passwd",
    "_private_key",
    "_access_token",
    "_refresh_token",
    "_auth_token",
    "_session_token",
    "_id_token",
)

# Env / argv flags: require word-ish secret markers, not bare "token" substring
# inside "max_tokens" / "tokenizer".
_SECRET_ENV_RE = re.compile(
    r"(^|_)(api[_-]?key|secret|password|passwd|authorization|bearer|"
    r"credential|private[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"auth[_-]?token|session[_-]?token|id[_-]?token)(_|$)|^token$",
    re.I,
)

# Value shapes that look like live credentials regardless of key name.
_SECRET_VALUE_RE = re.compile(
    r"(?i)^("
    r"sk-[A-Za-z0-9_\-]{8,}"  # OpenAI-style
    r"|or-[A-Za-z0-9_\-]{8,}"  # OpenRouter-ish
    r"|ghp_[A-Za-z0-9]{20,}"  # GitHub PAT
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"  # Slack
    r"|Bearer\s+\S{8,}"
    r")$"
)

_REDACTED = "***REDACTED***"


def _key_is_secret(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in _SECRET_KEY_EXACT:
        return True
    # OPENAI_API_KEY style env keys
    if lowered.endswith("_api_key") or lowered.endswith("_apikey"):
        return True
    return any(lowered.endswith(suf) for suf in _SECRET_KEY_SUFFIXES)


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
        key_s = str(key)
        if _key_is_secret(key_s) or key_s.lower() in extra:
            out[key_s] = _REDACTED
        elif isinstance(val, Mapping):
            out[key_s] = redact_mapping(val, extra_keys=extra_keys)
        elif isinstance(val, list):
            out[key_s] = [scrub_secrets(v) for v in val]
        elif isinstance(val, str) and _looks_like_secret_string(key_s, val):
            out[key_s] = _REDACTED
        else:
            out[key_s] = val
    return out


def _looks_like_secret_string(key: str, value: str) -> bool:
    if not value or value.lower() in {"ollama", "dummy", "test", "none", "n/a", ""}:
        return False
    # Key-name heuristic for env-style keys (exact token / api_key etc.).
    if _SECRET_ENV_RE.search(key) and value.lower() not in {
        "ollama",
        "dummy",
        "test",
        "none",
        "n/a",
    }:
        return True
    # Value shape (sk-..., Bearer ..., etc.) under any key.
    if _SECRET_VALUE_RE.match(value.strip()):
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
        if _SECRET_ENV_RE.search(key) or _key_is_secret(key):
            safe[key] = "set" if val else "unset"
        else:
            safe[key] = val
    return safe
