"""Unicode-aware tokenization shared by S2 (taint), S3 (intent), and T0 registry.

Scoring formulas / thresholds are unchanged — this only fixes ASCII-only
token extractors that dropped Cyrillic (and similar) into brittle char n-grams.
"""

from __future__ import annotations

import re
from typing import List, Set

# Word tokens (letters/digits/_ across scripts). Rich form also keeps path/email
# punctuation used by S2 attribution.
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_RICH_RE = re.compile(r"[\w@./:+-]+", re.UNICODE)

# Apostrophe / modifier-letter variants → ASCII apostrophe before casefold work.
_APOSTROPHE_TRANS = str.maketrans(
    {
        "\u2019": "'",  # ’
        "\u2018": "'",  # ‘
        "\u02bc": "'",  # ʼ
        "`": "'",
        "\u00b4": "'",
    }
)

# Longest-first Ukrainian inflectional endings (light stem; min stem length
# enforced below). Not a full morphological analyser — just enough for
# Києві≈Київ / погоду≈погода style parity with English.
_UK_ENDINGS = (
    "ього",
    "ьому",
    "іями",
    "ями",
    "ові",
    "еві",
    "єві",
    "ами",
    "ах",
    "ям",
    "ів",
    "їв",
    "ом",
    "ем",
    "єм",
    "ою",
    "ею",
    "єю",
    "ую",
    "юю",
    "ій",
    "їй",
    "ю",
    "я",
    "у",
    "і",
    "ї",
    "е",
    "є",
    "о",
    "и",
    "й",
    "а",
)

# Stem alternations after ending strip (Києві → києв → київ).
_UK_STEM_ALIASES = {
    "києв": "київ",
    "львов": "львів",
    "одес": "одеса",
}

_HAS_CYR = re.compile(r"[\u0400-\u04FF]")
_HAS_LAT = re.compile(r"[A-Za-z]")


def normalize_apostrophes(text: str) -> str:
    return (text or "").translate(_APOSTROPHE_TRANS)


def _light_stem(token: str) -> str:
    """Casefold + light UK/EN inflection strip. Preserves identifiers."""
    t = normalize_apostrophes(token).casefold()
    if not t:
        return t
    if t.endswith("'s") and len(t) > 3:
        t = t[:-2]
    elif t.endswith("'") and len(t) > 2:
        t = t[:-1]

    if _HAS_CYR.search(t):
        for end in _UK_ENDINGS:
            if t.endswith(end) and len(t) - len(end) >= 3:
                stem = t[: -len(end)]
                return _UK_STEM_ALIASES.get(stem, stem)
        return _UK_STEM_ALIASES.get(t, t)

    # Very light English: trailing plural -s / -es when the stem stays long.
    if _HAS_LAT.search(t) and not any(ch in t for ch in "@./:+-"):
        if t.endswith("es") and len(t) > 5 and t[-3] in "sxz" or (
            t.endswith("es") and len(t) > 5 and t[-3] == "h"
        ):
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss") and len(t) > 4:
            return t[:-1]
    return t


def tokenize(
    text: str,
    *,
    rich: bool = False,
    min_len: int = 1,
) -> List[str]:
    """
    Extract normalised Unicode tokens.

    ``rich=True`` keeps ``@./:+-`` (emails, paths, ISO-ish ids) for S2.
    """
    if not text:
        return []
    raw = normalize_apostrophes(str(text))
    pat = _RICH_RE if rich else _WORD_RE
    out: List[str] = []
    for m in pat.findall(raw):
        stem = _light_stem(m)
        if len(stem) >= min_len:
            out.append(stem)
    return out


def normalize_surface(text: str) -> str:
    """Casefold + apostrophe fix + per-word light stem for n-gram parity."""
    raw = normalize_apostrophes(str(text or ""))

    def _repl(m: re.Match) -> str:
        return _light_stem(m.group(0))

    # Stem alphabetic/underscore runs; leave digits/punct as-is.
    stemmed = _WORD_RE.sub(_repl, raw)
    return re.sub(r"\s+", " ", stemmed.casefold()).strip()


def char_ngrams(text: str, n: int = 3) -> Set[str]:
    s = normalize_surface(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}
