"""
Product naming for the MCP runtime guard.

Canonical names
---------------
* Ukrainian display: **Оберіг**
* ASCII / English: **Oberih**
* Machine identifier: ``oberih``

Legacy identifiers (``sentinel``, ``MCP-Sentinel``, ``mcp-sentinel``, …) remain
accepted as aliases so existing CLI flags, imports, result JSON, and datasets
keep working. Package path ``llm_tester.sentinel`` and interchange schema
``mcp-sentinel-trace/v1`` are stable wire formats — do not rewrite historical
run artefacts.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Sequence, Tuple

# Canonical product names
DISPLAY_NAME_UK = "Оберіг"
DISPLAY_NAME_EN = "Oberih"
DISPLAY_NAME = f"{DISPLAY_NAME_UK} ({DISPLAY_NAME_EN})"
MACHINE_ID = "oberih"

# Historical product / package labels (aliases only)
LEGACY_DISPLAY_NAME = "MCP-Sentinel"
LEGACY_MACHINE_ID = "sentinel"
LEGACY_PACKAGE = "llm_tester.sentinel"
LEGACY_TRACE_SCHEMA = "mcp-sentinel-trace/v1"

# Defense id aliases → canonical machine id for the full cascade.
_FULL_STACK_ALIASES: FrozenSet[str] = frozenset(
    {
        MACHINE_ID,
        LEGACY_MACHINE_ID,
        "mcp_sentinel",
        "mcp-sentinel",
        "mcp.sentinel",
    }
)
_NO_S4_ALIASES: FrozenSet[str] = frozenset(
    {
        f"{MACHINE_ID}_no_s4",
        f"{LEGACY_MACHINE_ID}_no_s4",
        "oberih-no-s4",
        "sentinel-no-s4",
    }
)
_S4_ONLY_ALIASES: FrozenSet[str] = frozenset({"s4_only", "s4-only", "only_s4"})


def normalize_defense_token(name: Optional[str]) -> str:
    """Lowercase / strip / unify hyphens for defense id matching."""
    return (name or "").lower().strip().replace("-", "_")


def is_oberih_defense(name: Optional[str]) -> bool:
    """True if ``name`` selects the Оберіг / Sentinel cascade (any variant)."""
    n = (name or "").lower().strip()
    n_us = n.replace("-", "_")
    if n in _FULL_STACK_ALIASES or n_us in {a.replace("-", "_") for a in _FULL_STACK_ALIASES}:
        return True
    if n in _NO_S4_ALIASES or n_us in {a.replace("-", "_") for a in _NO_S4_ALIASES}:
        return True
    if n in _S4_ONLY_ALIASES or n_us in {a.replace("-", "_") for a in _S4_ONLY_ALIASES}:
        return True
    return False


def is_full_stack_defense(name: Optional[str]) -> bool:
    n = (name or "").lower().strip()
    return n in _FULL_STACK_ALIASES or n.replace("-", "_") in {
        "oberih",
        "sentinel",
        "mcp_sentinel",
        "mcp.sentinel",
    }


def is_no_s4_defense(name: Optional[str]) -> bool:
    n = (name or "").lower().strip()
    n_us = n.replace("-", "_")
    return n in _NO_S4_ALIASES or n_us in {"oberih_no_s4", "sentinel_no_s4"}


def is_s4_only_defense(name: Optional[str]) -> bool:
    n = (name or "").lower().strip().replace("-", "_")
    return n in {"s4_only", "only_s4"}


def canonical_defense_id(name: Optional[str]) -> str:
    """
    Map a defense token to the canonical machine id for *new* labelling.

    Unknown names pass through unchanged (lowercased). Full-stack aliases map
    to ``oberih``; legacy ``sentinel_no_s4`` maps to ``oberih_no_s4``.
    """
    raw = (name or "").strip()
    n = raw.lower()
    if is_s4_only_defense(n):
        return "s4_only"
    if is_no_s4_defense(n):
        return f"{MACHINE_ID}_no_s4"
    if is_full_stack_defense(n):
        return MACHINE_ID
    if n in ("none", "off", "allow"):
        return "none"
    return n or "none"


def preserve_defense_label(name: Optional[str]) -> str:
    """
    Label written into result rows.

    Keeps the caller's token when it is a known alias (so ``--defense sentinel``
    still records ``\"defense\": \"sentinel\"`` for dataset continuity). Unknown
    / empty defaults to the canonical machine id.
    """
    raw = (name or "").strip()
    if not raw:
        return MACHINE_ID
    if is_oberih_defense(raw) or raw.lower() in ("none", "off", "allow"):
        return raw
    return raw


def find_oberih_defense_key(by_defense: dict) -> Optional[str]:
    """Return the key used in a results map for the Оберіг condition, if any."""
    if not by_defense:
        return None
    # Prefer canonical key when both legacy and new are present.
    for preferred in (MACHINE_ID, LEGACY_MACHINE_ID, f"{MACHINE_ID}_no_s4", f"{LEGACY_MACHINE_ID}_no_s4"):
        if preferred in by_defense:
            return preferred
    for key in by_defense:
        if is_oberih_defense(str(key)):
            return str(key)
    return None


def defense_cli_choices() -> Tuple[str, ...]:
    """CLI ``--defense`` / ``--baseline`` choices (canonical first, then legacy)."""
    return (
        MACHINE_ID,
        LEGACY_MACHINE_ID,
        f"{MACHINE_ID}_no_s4",
        f"{LEGACY_MACHINE_ID}_no_s4",
        "s4_only",
    )


def default_paired_defenses() -> Sequence[str]:
    """Default none vs Оберіг matrix (canonical machine ids)."""
    return ("none", MACHINE_ID)


def product_blurb(*, uk: bool = False) -> str:
    if uk:
        return (
            f"{DISPLAY_NAME_UK} (лат. {DISPLAY_NAME_EN}; машинний id ``{MACHINE_ID}``) "
            f"— колишній {LEGACY_DISPLAY_NAME}; "
            f"аліаси ``{LEGACY_MACHINE_ID}`` / CLI ``sentinel-proxy`` збережено."
        )
    return (
        f"{DISPLAY_NAME} — cascaded provenance-aware MCP runtime guard "
        f"(formerly {LEGACY_DISPLAY_NAME}; machine id ``{MACHINE_ID}``, "
        f"legacy alias ``{LEGACY_MACHINE_ID}``)."
    )


def known_defense_help() -> str:
    return (
        f"{MACHINE_ID}|{LEGACY_MACHINE_ID}|{MACHINE_ID}_no_s4|"
        f"{LEGACY_MACHINE_ID}_no_s4|s4_only|none"
    )


def expand_defense_aliases(names: Iterable[str]) -> Tuple[str, ...]:
    """Deduplicate while accepting mixed oberih/sentinel lists."""
    out = []
    seen = set()
    for n in names:
        label = preserve_defense_label(n)
        if label not in seen:
            seen.add(label)
            out.append(label)
    return tuple(out)
