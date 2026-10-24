"""Threshold calibration on a train/dev split; evaluate on a held-out split."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_tester.agent.eval import EvalMetrics, evaluate_traces, format_metrics
from llm_tester.agent.traces import TraceStep
from llm_tester.sentinel.guard import build_sentinel
from llm_tester.sentinel.policy import SentinelPolicy, merge_policy

# Scalar held-out metrics aggregated across seeds.
_CI_METRIC_KEYS = (
    "tpr",
    "fpr",
    "f1",
    "precision",
    "recall",
    "asr_with_guard",
    "asr_reduction",
    "escalation_rate",
)


@dataclass
class SplitSpec:
    strategy: str
    train_frac: float
    seed: int
    train_ids: List[str] = field(default_factory=list)
    heldout_ids: List[str] = field(default_factory=list)
    group_key: str = "tool"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _group_key(trace: TraceStep, strategy: str) -> str:
    """Group for stratified split — by attacker tool or attack_type."""
    key = (strategy or "tool").lower()
    if key in ("attack_type", "attack", "type"):
        return str(trace.attack_type or "other")
    if key in ("tool", "attacker_tool"):
        if trace.call is not None:
            return str(trace.call.name or "unknown_tool")
        meta_tool = (trace.meta or {}).get("user_tool") or (trace.meta or {}).get(
            "attacker_tool"
        )
        if meta_tool:
            return str(meta_tool)
        if trace.tools:
            return str(trace.tools[0].name)
        return "no_tool"
    if key in ("id_hash", "hash"):
        digest = hashlib.sha256(trace.id.encode("utf-8")).hexdigest()
        return digest[:2]
    return str(trace.attack_type or "other")


def split_traces(
    traces: Sequence[TraceStep],
    *,
    strategy: str = "tool",
    train_frac: float = 0.5,
    seed: int = 0,
) -> Tuple[List[TraceStep], List[TraceStep], SplitSpec]:
    """
    Split traces into train/dev and held-out by grouping key.

    Groups (tools / attack types) are assigned wholly to one side so the
    held-out set does not leak the same attacker tool into calibration.
    """
    if not 0.05 <= train_frac <= 0.95:
        raise ValueError("train_frac must be in [0.05, 0.95]")

    rng = random.Random(seed)
    groups: Dict[str, List[TraceStep]] = {}
    for t in traces:
        groups.setdefault(_group_key(t, strategy), []).append(t)

    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_train_groups = max(1, int(round(len(keys) * train_frac)))
    if n_train_groups >= len(keys) and len(keys) > 1:
        n_train_groups = len(keys) - 1
    train_keys = set(keys[:n_train_groups])
    held_keys = set(keys[n_train_groups:]) or set(keys[-1:])

    train: List[TraceStep] = []
    held: List[TraceStep] = []
    for k, items in groups.items():
        if k in train_keys and k not in held_keys:
            train.extend(items)
        else:
            held.extend(items)

    # Ensure neither side is empty on tiny sets.
    if not train and held:
        train = held[: max(1, len(held) // 2)]
        held = held[len(train) :]
    if not held and train:
        held = train[-max(1, len(train) // 4) :]
        train = train[: len(train) - len(held)]

    spec = SplitSpec(
        strategy=strategy,
        train_frac=train_frac,
        seed=seed,
        train_ids=[t.id for t in train],
        heldout_ids=[t.id for t in held],
        group_key=strategy,
    )
    return train, held, spec


def _score_for_objective(metrics: EvalMetrics, objective: str) -> float:
    obj = (objective or "f1").lower()
    ai = metrics.any_intervention
    bo = metrics.block_only
    if obj in ("f1", "any_f1"):
        return float(ai.f1 or 0.0)
    if obj in ("block_f1",):
        return float(bo.f1 or 0.0)
    if obj in ("tpr", "any_tpr"):
        return float(ai.tpr or 0.0)
    if obj in ("youden", "j"):
        # Youden's J = TPR - FPR (prefer high detection, low FP).
        return float((ai.tpr or 0.0) - (ai.fpr or 0.0))
    if obj in ("f1_fpr",):
        # Penalise FPR lightly.
        return float((ai.f1 or 0.0) - 0.5 * (ai.fpr or 0.0))
    return float(ai.f1 or 0.0)


def calibrate_thresholds(
    train: Sequence[TraceStep],
    base_policy: Optional[SentinelPolicy] = None,
    *,
    objective: str = "youden",
    tau_low_grid: Optional[Sequence[float]] = None,
    tau_high_grid: Optional[Sequence[float]] = None,
    max_fpr: Optional[float] = 0.15,
) -> Tuple[SentinelPolicy, EvalMetrics, Dict[str, Any]]:
    """
    Grid-search ``tau_low`` / ``tau_high`` on the train split.

    Returns (best_policy, train_metrics, search_meta).
    """
    base = base_policy or SentinelPolicy(enable_s4=False)
    lows = list(tau_low_grid or (0.15, 0.20, 0.25, 0.30, 0.35))
    highs = list(tau_high_grid or (0.55, 0.60, 0.65, 0.70, 0.75, 0.80))

    best_score = -1e9
    best_policy = base
    best_metrics: Optional[EvalMetrics] = None
    trials: List[Dict[str, Any]] = []

    for lo, hi in itertools.product(lows, highs):
        if hi <= lo + 0.05:
            continue
        pol = merge_policy(base, {"tau_low": lo, "tau_high": hi, "enable_s4": False})
        metrics = evaluate_traces(train, build_sentinel(policy=pol))
        score = _score_for_objective(metrics, objective)
        fpr = metrics.any_intervention.fpr
        trial = {
            "tau_low": lo,
            "tau_high": hi,
            "score": score,
            "tpr": metrics.any_intervention.tpr,
            "fpr": fpr,
            "f1": metrics.any_intervention.f1,
        }
        trials.append(trial)
        if max_fpr is not None and fpr is not None and fpr > max_fpr:
            continue
        if score > best_score:
            best_score = score
            best_policy = pol
            best_metrics = metrics

    if best_metrics is None:
        # Relax FPR constraint if nothing feasible.
        for trial in sorted(trials, key=lambda t: t["score"], reverse=True):
            pol = merge_policy(
                base,
                {
                    "tau_low": trial["tau_low"],
                    "tau_high": trial["tau_high"],
                    "enable_s4": False,
                },
            )
            best_policy = pol
            best_metrics = evaluate_traces(train, build_sentinel(policy=pol))
            best_score = trial["score"]
            break

    assert best_metrics is not None
    meta = {
        "objective": objective,
        "max_fpr": max_fpr,
        "best_score": best_score,
        "best_tau_low": best_policy.tau_low,
        "best_tau_high": best_policy.tau_high,
        "n_trials": len(trials),
        "trials": trials,
    }
    return best_policy, best_metrics, meta


def run_calibration(
    traces: Sequence[TraceStep],
    *,
    base_policy: Optional[SentinelPolicy] = None,
    strategy: str = "tool",
    train_frac: float = 0.5,
    seed: int = 0,
    objective: str = "youden",
    max_fpr: Optional[float] = 0.15,
) -> Dict[str, Any]:
    """Full calibrate-on-train / evaluate-on-heldout pipeline."""
    train, held, split = split_traces(
        traces, strategy=strategy, train_frac=train_frac, seed=seed
    )
    policy, train_metrics, search = calibrate_thresholds(
        train, base_policy, objective=objective, max_fpr=max_fpr
    )
    held_metrics = evaluate_traces(held, build_sentinel(policy=policy))
    return {
        "split": split.to_dict(),
        "calibrated_policy": {
            "tau_low": policy.tau_low,
            "tau_high": policy.tau_high,
            "risk_aggregation": policy.risk_aggregation,
        },
        "search": {
            k: v for k, v in search.items() if k != "trials"
        },
        "train_metrics": train_metrics.to_dict(),
        "heldout_metrics": held_metrics.to_dict(),
        "train_report": format_metrics(train_metrics, title="calibrate/train"),
        "heldout_report": format_metrics(held_metrics, title="calibrate/heldout"),
    }


def write_calibration_report(result: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: v
        for k, v in result.items()
        if k not in ("train_report", "heldout_report")
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def _nested_get(d: Dict[str, Any], *keys: str) -> Optional[float]:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def _mean_ci95(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Sample mean and 95% CI (normal approx; t for small n via 1.96≈z)."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    n = len(xs)
    mean = statistics.mean(xs)
    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }
    std = statistics.stdev(xs)
    # Student-t critical values for common small n (two-sided 95%).
    t_crit = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }.get(n, 1.96)
    half = t_crit * std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def run_calibration_multi_seed(
    traces: Sequence[TraceStep],
    *,
    seeds: Sequence[int],
    base_policy: Optional[SentinelPolicy] = None,
    strategy: str = "tool",
    train_frac: float = 0.5,
    objective: str = "youden",
    max_fpr: Optional[float] = 0.15,
) -> Dict[str, Any]:
    """
    Repeat calibration over ``seeds``; report per-seed results plus mean ± 95% CI
    on held-out any-intervention metrics.
    """
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("seeds must be non-empty")

    per_seed: List[Dict[str, Any]] = []
    for seed in seed_list:
        one = run_calibration(
            traces,
            base_policy=base_policy,
            strategy=strategy,
            train_frac=train_frac,
            seed=seed,
            objective=objective,
            max_fpr=max_fpr,
        )
        per_seed.append(one)

    # Aggregate held-out scalars (prefer any_intervention.* when present).
    aggregate: Dict[str, Any] = {}
    for key in _CI_METRIC_KEYS:
        vals = []
        for r in per_seed:
            held = r.get("heldout_metrics") or {}
            v = _nested_get(held, "any_intervention", key)
            if v is None:
                v = _nested_get(held, key)
            if v is not None:
                vals.append(v)
        aggregate[key] = _mean_ci95(vals)

    # Also aggregate calibrated thresholds.
    tau_lows = [
        float((r.get("calibrated_policy") or {}).get("tau_low"))
        for r in per_seed
        if (r.get("calibrated_policy") or {}).get("tau_low") is not None
    ]
    tau_highs = [
        float((r.get("calibrated_policy") or {}).get("tau_high"))
        for r in per_seed
        if (r.get("calibrated_policy") or {}).get("tau_high") is not None
    ]
    aggregate["tau_low"] = _mean_ci95(tau_lows)
    aggregate["tau_high"] = _mean_ci95(tau_highs)

    # Representative single-seed payload (first seed) for back-compat fields.
    primary = per_seed[0]
    return {
        "seeds": seed_list,
        "n_seeds": len(seed_list),
        "strategy": strategy,
        "train_frac": train_frac,
        "objective": objective,
        "per_seed": [
            {
                k: v
                for k, v in r.items()
                if k not in ("train_report", "heldout_report")
            }
            for r in per_seed
        ],
        "heldout_aggregate": aggregate,
        "split": primary["split"],
        "calibrated_policy": primary["calibrated_policy"],
        "search": primary.get("search"),
        "train_metrics": primary["train_metrics"],
        "heldout_metrics": primary["heldout_metrics"],
        "train_report": primary["train_report"],
        "heldout_report": primary["heldout_report"],
        "aggregate_report": format_aggregate(aggregate, seeds=seed_list),
    }


def format_aggregate(
    aggregate: Dict[str, Any], *, seeds: Sequence[int]
) -> str:
    lines = [
        f"=== calibrate/heldout aggregate over seeds={list(seeds)} ===",
    ]
    for key, stats in aggregate.items():
        if not isinstance(stats, dict) or stats.get("mean") is None:
            continue
        lines.append(
            f"{key}: mean={stats['mean']:.4f} "
            f"95%CI=[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}] "
            f"(n={stats['n']}, std={stats['std']:.4f})"
        )
    return "\n".join(lines)


def parse_seeds_arg(
    seeds: Optional[str] = None,
    *,
    seed: int = 0,
    n_seeds: Optional[int] = None,
) -> List[int]:
    """
    Resolve seed list from CLI flags.

    - ``--seeds 0,1,2`` → explicit list
    - ``--n-seeds 5 --seed 0`` → ``0..4`` starting at ``seed``
    - default → ``[seed]``
    """
    if seeds:
        out = []
        for part in str(seeds).split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        if not out:
            raise ValueError("--seeds produced an empty list")
        return out
    if n_seeds is not None:
        if n_seeds < 1:
            raise ValueError("--n-seeds must be >= 1")
        return list(range(int(seed), int(seed) + int(n_seeds)))
    return [int(seed)]
