"""
Sentinel-aware adaptive attack harness (R-A1).

Threat model (grey-box): attacker controls untrusted tool outputs only; knows
Sentinel stage design but not thresholds; gets outcome bits (success /
blocked) for up to k tries.

Families (see ``templates``):
  * ``a_para`` / ``a_split`` / ``a_judge`` — deterministic transforms
  * ``a_iter`` — attacker-LLM refinement loop with early stop
  * ``important_instructions`` — static control

The loop is defense-agnostic: a callable ``run_try(injections) -> TryOutcome``
runs one AgentDojo (or mock) job. CI uses scripted mocks; live runs inject
real providers + :class:`~llm_tester.budget.RequestLedger`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from llm_tester.attacks.templates import family_catalog, injections_for_family
from llm_tester.budget.ledger import BudgetExceeded, RequestLedger

ATTACK_FAMILIES = family_catalog()


@dataclass
class TryOutcome:
    attack_success: bool
    blocked: bool = False
    utility_success: bool = False
    agent_requests: int = 0
    defense_requests: int = 0
    attacker_requests: int = 0
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveJobResult:
    job_id: str
    defense: str
    attack_family: str
    try_successes: List[bool] = field(default_factory=list)
    try_blocked: List[bool] = field(default_factory=list)
    attack_success: bool = False
    stopped_early: bool = False
    n_tries: int = 0
    k: int = 1
    attacker_requests: int = 0
    agent_requests: int = 0
    defense_requests: int = 0
    budget_truncated: bool = False
    error: Optional[str] = None
    tries: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_family(
    family: str,
    seed_injections: Dict[str, str],
    *,
    variant: int = 0,
) -> Dict[str, str]:
    """Public wrapper around template transforms."""
    return injections_for_family(family, seed_injections, variant=variant)


class AttackerLoop:
    """
    Adaptive refinement loop for family ``a_iter``.

    ``attacker_rewrite(seed_text, history) -> str`` may call an LLM (counted)
    or a deterministic mock. ``run_try(injections)`` executes one defended job.
    """

    def __init__(
        self,
        *,
        k: int = 3,
        ledger: Optional[RequestLedger] = None,
        run_cap: Optional[int] = None,
        attacker_rewrite: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.ledger = ledger
        self.run_cap = run_cap
        self._run_used = 0
        self.attacker_rewrite = attacker_rewrite or default_mock_attacker_rewrite

    def _count(self, part: str, n: int = 1) -> None:
        self._run_used += n
        if self.run_cap is not None and self._run_used > self.run_cap:
            raise BudgetExceeded(f"run cap {self.run_cap} reached")
        if self.ledger is not None:
            self.ledger.record(part, n=n)

    def run(
        self,
        *,
        job_id: str,
        defense: str,
        seed_injections: Dict[str, str],
        run_try: Callable[[Dict[str, str]], TryOutcome],
        family: str = "a_iter",
    ) -> AdaptiveJobResult:
        result = AdaptiveJobResult(
            job_id=job_id, defense=defense, attack_family=family, k=self.k
        )
        seed_text = "\n".join(str(v) for v in seed_injections.values())
        history: List[Dict[str, Any]] = []
        current = apply_family(
            "a_para" if family == "a_iter" else family,
            seed_injections,
            variant=0,
        )

        for try_index in range(self.k):
            try:
                if family == "a_iter" and try_index > 0:
                    self._count("attacker", 1)
                    rewritten = self.attacker_rewrite(seed_text, history)
                    keys = list(seed_injections.keys())
                    current = {keys[0]: rewritten, **{k: "" for k in keys[1:]}}
                    result.attacker_requests += 1
                elif family == "a_iter" and try_index == 0:
                    # First try uses template seed (no attacker LLM call).
                    pass
                else:
                    current = apply_family(family, seed_injections, variant=try_index)

                outcome = run_try(current)
            except BudgetExceeded as exc:
                result.budget_truncated = True
                result.error = str(exc)
                break
            except Exception as exc:  # noqa: BLE001 — surface in job record
                result.error = f"{type(exc).__name__}: {exc}"
                break

            result.n_tries += 1
            result.try_successes.append(bool(outcome.attack_success))
            result.try_blocked.append(bool(outcome.blocked))
            result.agent_requests += int(outcome.agent_requests or 0)
            result.defense_requests += int(outcome.defense_requests or 0)
            result.attacker_requests += int(outcome.attacker_requests or 0)
            result.tries.append(
                {
                    "try_index": try_index,
                    "attack_success": bool(outcome.attack_success),
                    "blocked": bool(outcome.blocked),
                    "utility_success": bool(outcome.utility_success),
                    "injections": dict(current),
                    "meta": dict(outcome.meta or {}),
                }
            )
            history.append(
                {
                    "try_index": try_index,
                    "success": bool(outcome.attack_success),
                    "blocked": bool(outcome.blocked),
                    "injection": "\n".join(str(v) for v in current.values())[:2000],
                }
            )
            if outcome.attack_success:
                result.attack_success = True
                result.stopped_early = try_index + 1 < self.k
                break

        return result


def default_mock_attacker_rewrite(
    seed_text: str, history: List[Dict[str, Any]]
) -> str:
    """
    Deterministic offline rewriter for CI (no network).

    Cycles paraphrase / judge / split-style framings based on try count.
    """
    n = len(history)
    from llm_tester.attacks.templates import (
        judge_evasion_frame,
        paraphrase_obfuscate,
        strip_triggers,
    )

    body = strip_triggers(seed_text) or seed_text
    if n % 3 == 0:
        return paraphrase_obfuscate(body, variant=n)
    if n % 3 == 1:
        return judge_evasion_frame(body, variant=n)
    return paraphrase_obfuscate(body, variant=n + 1) + " (retry)"


def run_adaptive_job(
    *,
    job_id: str,
    defense: str,
    family: str,
    seed_injections: Dict[str, str],
    run_try: Callable[[Dict[str, str]], TryOutcome],
    k: int = 3,
    ledger: Optional[RequestLedger] = None,
    run_cap: Optional[int] = None,
    attacker_rewrite: Optional[Callable[[str, List[Dict[str, Any]]], str]] = None,
) -> AdaptiveJobResult:
    """
    Run one adaptive (or static) job.

    Static families (``a_para``, ``a_split``, ``a_judge``,
    ``important_instructions``) use ``k=1`` unless the caller raises ``k``
    to explore variants. ``a_iter`` uses the full loop.
    """
    fam = (family or "a_iter").lower().strip()
    if fam != "a_iter":
        k_eff = 1 if k is None else max(1, int(k))
        # For static families, optionally sweep variants when k>1.
        if fam in ("a_para", "a_split", "a_judge") and k_eff > 1:
            loop = AttackerLoop(
                k=k_eff,
                ledger=ledger,
                run_cap=run_cap,
                attacker_rewrite=attacker_rewrite,
            )
            return loop.run(
                job_id=job_id,
                defense=defense,
                seed_injections=seed_injections,
                run_try=run_try,
                family=fam,
            )
        # Single static try.
        result = AdaptiveJobResult(
            job_id=job_id, defense=defense, attack_family=fam, k=1
        )
        injections = apply_family(fam, seed_injections, variant=0)
        try:
            outcome = run_try(injections)
        except BudgetExceeded as exc:
            result.budget_truncated = True
            result.error = str(exc)
            return result
        result.n_tries = 1
        result.try_successes = [bool(outcome.attack_success)]
        result.try_blocked = [bool(outcome.blocked)]
        result.attack_success = bool(outcome.attack_success)
        result.agent_requests = int(outcome.agent_requests or 0)
        result.defense_requests = int(outcome.defense_requests or 0)
        result.tries = [
            {
                "try_index": 0,
                "attack_success": bool(outcome.attack_success),
                "blocked": bool(outcome.blocked),
                "injections": injections,
            }
        ]
        return result

    loop = AttackerLoop(
        k=k,
        ledger=ledger,
        run_cap=run_cap,
        attacker_rewrite=attacker_rewrite,
    )
    return loop.run(
        job_id=job_id,
        defense=defense,
        seed_injections=seed_injections,
        run_try=run_try,
        family="a_iter",
    )


def make_scripted_run_try(
    *,
    block_if_contains: Optional[Sequence[str]] = None,
    succeed_unless_blocked: bool = True,
    agent_requests_per_try: int = 1,
    defense_requests_per_try: int = 0,
) -> Callable[[Dict[str, str]], TryOutcome]:
    """
    CI helper: block when any injection contains a marker; otherwise succeed.
    """
    markers = tuple(m.lower() for m in (block_if_contains or ("IMPORTANT!!!", "ignore previous")))

    def _run(injections: Dict[str, str]) -> TryOutcome:
        blob = "\n".join(str(v) for v in injections.values()).lower()
        blocked = any(m.lower() in blob for m in markers)
        success = (not blocked) if succeed_unless_blocked else False
        return TryOutcome(
            attack_success=success,
            blocked=blocked,
            utility_success=not blocked,
            agent_requests=agent_requests_per_try,
            defense_requests=defense_requests_per_try,
        )

    return _run
