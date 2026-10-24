"""S2 — argument attribution / taint (core novelty).

For each call argument a^j:
  π(a^j) = |tok(a^j) ∩ tok(O_<t) \\ tok(u)| / |tok(a^j)|

Sensitive sink arguments with high untrusted attribution raise risk.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from llm_tester.sentinel.policy import SentinelPolicy, is_sensitive_arg
from llm_tester.sentinel.types import StageScore, ToolCall

_TOKEN_RE = re.compile(r"[A-Za-z0-9_@./:+-]{2,}")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(str(text))]


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    s = re.sub(r"\s+", " ", str(text).lower()).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


@dataclass
class TaintedSource:
    source: str
    content: str
    tokens: Set[str] = field(default_factory=set)
    ngrams: Set[str] = field(default_factory=set)


@dataclass
class TaintStore:
    """Provenance store: untrusted tool outputs and trusted user request."""

    user_request: str = ""
    user_tokens: Set[str] = field(default_factory=set)
    user_ngrams: Set[str] = field(default_factory=set)
    sources: List[TaintedSource] = field(default_factory=list)
    n: int = 3

    def set_user_task(self, text: str) -> None:
        self.user_request = text or ""
        self.user_tokens = set(tokenize(self.user_request))
        self.user_ngrams = char_ngrams(self.user_request, self.n)

    def add_tool_output(self, source: str, content: str) -> TaintedSource:
        tokens = set(tokenize(content))
        grams = char_ngrams(content, self.n)
        entry = TaintedSource(
            source=source, content=content, tokens=tokens, ngrams=grams
        )
        self.sources.append(entry)
        return entry

    def clear_outputs(self) -> None:
        self.sources.clear()

    @property
    def untrusted_tokens(self) -> Set[str]:
        out: Set[str] = set()
        for s in self.sources:
            out |= s.tokens
        return out

    @property
    def untrusted_ngrams(self) -> Set[str]:
        out: Set[str] = set()
        for s in self.sources:
            out |= s.ngrams
        return out


def argument_attribution(
    value: str,
    store: TaintStore,
    *,
    use_ngrams: bool = True,
) -> Tuple[float, Dict]:
    """
    Fraction of argument tokens that appear in untrusted outputs but NOT in u.

    Combines token overlap with optional char-n-gram overlap for short values
    (emails, IBANs, guest ids) where tokenization alone is brittle.
    """
    tokens = tokenize(value)
    details: Dict = {"tokens": tokens, "tainted_tokens": [], "method": "tokens"}

    if not store.sources:
        return 0.0, details

    untrusted = store.untrusted_tokens
    user = store.user_tokens
    tainted = [t for t in tokens if t in untrusted and t not in user]

    token_pi = (len(tainted) / len(tokens)) if tokens else 0.0
    details["tainted_tokens"] = tainted
    details["token_pi"] = token_pi

    ngram_pi = 0.0
    if use_ngrams:
        grams = char_ngrams(value, store.n)
        if grams:
            u_grams = store.untrusted_ngrams
            user_grams = store.user_ngrams
            tainted_g = {g for g in grams if g in u_grams and g not in user_grams}
            ngram_pi = len(tainted_g) / len(grams)
            details["ngram_pi"] = ngram_pi
            details["tainted_ngram_count"] = len(tainted_g)
            details["method"] = "tokens+ngrams"

    # Max of the two signals; short values rely more on n-grams.
    if len(tokens) <= 2 and use_ngrams:
        pi = max(token_pi, ngram_pi)
    else:
        pi = max(token_pi, 0.5 * ngram_pi) if use_ngrams else token_pi

    details["pi"] = pi
    return float(min(1.0, pi)), details


class TaintStage:
    name = "S2"

    def __init__(
        self,
        policy: Optional[SentinelPolicy] = None,
        store: Optional[TaintStore] = None,
    ) -> None:
        self.policy = policy or SentinelPolicy()
        self.store = store or TaintStore(n=self.policy.taint_ngram_n)

    def bind_store(self, store: TaintStore) -> None:
        self.store = store

    def score_call(self, call: ToolCall) -> StageScore:
        start = time.perf_counter()
        reasons: List[str] = []
        per_arg: Dict[str, Dict] = {}
        weighted_scores: List[float] = []

        tool_l = call.name.lower()
        side_effect = any(
            sub.lower() in tool_l for sub in self.policy.side_effect_tools
        )

        for key, raw in (call.arguments or {}).items():
            value = raw if isinstance(raw, str) else str(raw)
            pi, detail = argument_attribution(value, self.store)
            sensitive = is_sensitive_arg(key, self.policy.sensitive_args)
            weight = (
                self.policy.sensitive_arg_weight
                if sensitive
                else self.policy.default_arg_weight
            )
            # Non-sink taint on a side-effect tool is more serious.
            if not sensitive and side_effect and pi >= 0.4:
                weight = max(weight, 0.9)
            weighted = min(1.0, pi * weight)
            per_arg[key] = {
                **detail,
                "sensitive": sensitive,
                "side_effect_tool": side_effect,
                "weight": weight,
                "weighted": weighted,
            }
            weighted_scores.append(weighted)
            if pi >= 0.3 and (sensitive or side_effect or weighted >= 0.4):
                tag = "sensitive" if sensitive else "arg"
                reasons.append(f"s2:{tag}_taint:{key}:{pi:.2f}")

        score = max(weighted_scores) if weighted_scores else 0.0
        # Attacker tool name drawn from untrusted output (and absent from u).
        name_pi, name_detail = argument_attribution(call.name, self.store)
        user_toks = {t.lower() for t in tokenize(self.store.user_request)}
        name_tokens = set(tokenize(call.name))
        name_in_user = bool(name_tokens & user_toks)
        if name_pi >= 0.4 and not name_in_user:
            score = max(score, min(1.0, 0.45 + 0.45 * name_pi))
            reasons.append(f"s2:tool_name_taint:{name_pi:.2f}")
            per_arg["__tool_name__"] = name_detail
        elif side_effect and score >= 0.25:
            # Mild boost when a side-effect tool is invoked after untrusted I/O.
            score = min(1.0, score + 0.1)
            reasons.append("s2:side_effect_tool")

        elapsed = (time.perf_counter() - start) * 1000
        return StageScore(
            stage=self.name,
            score=float(min(1.0, score)),
            reasons=reasons,
            details={"args": per_arg, "n_sources": len(self.store.sources)},
            latency_ms=elapsed,
        )

    def mark_output(self, source: str, content: str) -> None:
        self.store.add_tool_output(source, content)


def build_store_from_outputs(
    user_request: str,
    outputs: Iterable[Dict],
    *,
    n: int = 3,
) -> TaintStore:
    store = TaintStore(n=n)
    store.set_user_task(user_request)
    for o in outputs:
        store.add_tool_output(
            str(o.get("source", "tool")),
            str(o.get("content", "")),
        )
    return store
