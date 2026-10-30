"""Offline statistical helpers for adaptive / defense comparisons (no sklearn)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


def wilson_ci(successes: int, n: int, *, z: float = 1.96) -> Dict[str, Optional[float]]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n <= 0:
        return {"n": 0, "p": None, "low": None, "high": None}
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return {
        "n": n,
        "successes": successes,
        "p": p,
        "low": max(0.0, (centre - margin) / denom),
        "high": min(1.0, (centre + margin) / denom),
    }


def asr_at_1(outcomes: Sequence[bool]) -> Dict[str, Any]:
    """ASR@1: first-try attack success rate with Wilson CI."""
    n = len(outcomes)
    s = sum(1 for x in outcomes if x)
    ci = wilson_ci(s, n)
    return {"metric": "ASR@1", **ci}


def asr_at_k(try_successes: Sequence[Sequence[bool]]) -> Dict[str, Any]:
    """
    ASR@k: per-job success if **any** of the ≤k tries succeeded.

    ``try_successes[i]`` is the list of per-try success bits for job i
    (length ≤ k; early stop may shorten).
    """
    any_success = [any(tries) for tries in try_successes]
    return {"metric": "ASR@k", **wilson_ci(sum(any_success), len(any_success))}


def stopped_k_of_n(
    *,
    undefended_success_ids: Sequence[str],
    defended_success_ids: Sequence[str],
) -> Dict[str, Any]:
    """
    Of the n jobs where the attack succeeds with no defense, how many the
    defense still stops after k adaptive tries.
    """
    base = set(undefended_success_ids)
    still = set(defended_success_ids) & base
    stopped = base - still
    n = len(base)
    k = len(stopped)
    return {
        "metric": "stopped_k_of_n",
        "n_undefended_success": n,
        "k_stopped": k,
        "still_succeed_with_defense": len(still),
        "rate": (k / n) if n else None,
        "wilson": wilson_ci(k, n),
    }


def mcnemar_test(
    a_success: Sequence[bool],
    b_success: Sequence[bool],
) -> Dict[str, Any]:
    """
    McNemar test on paired binary outcomes (same jobs).

    Contigency: b01 = A fail & B success, b10 = A success & B fail.
    Uses continuity-corrected χ²; returns p-value via survival of χ²₁.
    """
    if len(a_success) != len(b_success):
        raise ValueError("McNemar requires equal-length paired sequences")
    b01 = b10 = 0
    for a, b in zip(a_success, b_success):
        if (not a) and b:
            b01 += 1
        elif a and (not b):
            b10 += 1
    discordant = b01 + b10
    if discordant == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / discordant
        # χ²₁ survival: erfc(sqrt(x/2))
        p = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "metric": "mcnemar",
        "b01": b01,
        "b10": b10,
        "chi2": chi2,
        "p_value": p,
        "n": len(a_success),
    }


def summarize_adaptive_results(
    jobs: Sequence[Dict[str, Any]],
    *,
    defense_a: str = "none",
    defense_b: str = "sentinel",
) -> Dict[str, Any]:
    """
    Aggregate a list of per-job result dicts produced by the adaptive runner.

    Expected keys per job: ``job_id``, ``defense``, ``try_successes`` (list[bool]),
    ``attack_success`` (bool, any-of-k), ``blocked`` (optional).
    """
    by_def: Dict[str, List[Dict[str, Any]]] = {}
    for j in jobs:
        by_def.setdefault(str(j.get("defense") or "none"), []).append(j)

    out: Dict[str, Any] = {"by_defense": {}}
    for name, rows in by_def.items():
        tries = [list(r.get("try_successes") or [bool(r.get("attack_success"))]) for r in rows]
        first = [bool(t[0]) if t else False for t in tries]
        out["by_defense"][name] = {
            "n": len(rows),
            "ASR@1": asr_at_1(first),
            "ASR@k": asr_at_k(tries),
            "attacker_requests_total": sum(int(r.get("attacker_requests") or 0) for r in rows),
            "agent_requests_total": sum(int(r.get("agent_requests") or 0) for r in rows),
            "defense_requests_total": sum(int(r.get("defense_requests") or 0) for r in rows),
        }

    # Paired McNemar when both defenses present on the same job ids.
    a_rows = {str(r.get("job_id")): r for r in by_def.get(defense_a, [])}
    b_rows = {str(r.get("job_id")): r for r in by_def.get(defense_b, [])}
    shared = sorted(set(a_rows) & set(b_rows))
    if shared:
        a_succ = [bool(a_rows[i].get("attack_success")) for i in shared]
        b_succ = [bool(b_rows[i].get("attack_success")) for i in shared]
        out["mcnemar"] = {
            "defense_a": defense_a,
            "defense_b": defense_b,
            **mcnemar_test(a_succ, b_succ),
        }
        # stopped_k_of_n using job ids
        undef_ids = [i for i in shared if bool(a_rows[i].get("attack_success"))]
        def_succ_ids = [i for i in shared if bool(b_rows[i].get("attack_success"))]
        out["stopped_k_of_n"] = stopped_k_of_n(
            undefended_success_ids=undef_ids,
            defended_success_ids=def_succ_ids,
        )
    return out
