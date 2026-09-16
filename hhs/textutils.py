"""Word counting and content-overlap metrics.

A word is a maximal span of Unicode word characters. The same definition is
applied to calibration texts, generated outputs, and requested counts, so the
ratio distribution and the reported word counts are measured on one scale.
"""

from __future__ import annotations

import re
from collections import Counter

WORD_RE = re.compile(r"\w+", re.UNICODE)


def words(text: str) -> list[str]:
    """Maximal spans of Unicode word characters."""
    return WORD_RE.findall(str(text))


def word_count(text: str) -> int:
    return len(words(text))


def compliant(realized: int, requested: float, tolerance: float = 0.2) -> int:
    """1 when the realized word count is within ``tolerance`` of the request."""
    return int(abs(realized - requested) <= tolerance * requested)


def unigram_f1(pred: str, ref: str) -> float:
    """Token-level F1 between two texts, lowercased."""
    p = Counter(w.lower() for w in words(pred))
    r = Counter(w.lower() for w in words(ref))
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(pred: str, ref: str) -> float:
    """ROUGE-L F1 over lowercased word sequences."""
    a = [w.lower() for w in words(pred)]
    b = [w.lower() for w in words(ref)]
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    if lcs == 0:
        return 0.0
    precision = lcs / len(a)
    recall = lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def repeated_fourgram_rate(text: str) -> float:
    """Fraction of 4-grams that are not unique."""
    toks = [w.lower() for w in words(text)]
    if len(toks) < 4:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def ends_with_terminal_punctuation(text: str) -> int:
    t = str(text).strip()
    return int(bool(t) and t[-1] in ".!?")


def complete_sentence(text: str) -> int:
    """Heuristic sentence completeness: ends in terminal punctuation and has
    at least a few words."""
    t = str(text).strip()
    if not t or t[-1] not in ".!?":
        return 0
    return int(len(words(t)) >= 2)
