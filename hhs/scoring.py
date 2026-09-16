"""Task correctness and length-constraint scoring.

Task accuracy is scored from the generated text alone, so every method is judged
under identical prompts and token budgets.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Sequence

from .textutils import unigram_f1, words

NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

KIND_NUMBER = "number"
KIND_YESNO = "yesno"
KIND_CHOICE = "choice"


def gsm8k_gold(answer_field: str) -> Optional[str]:
    """Final numeric answer of a reference of the form '... #### 42'."""
    tail = str(answer_field).split("####")[-1]
    match = NUMBER_RE.search(tail.replace(",", ""))
    return match.group(0) if match else None


def _as_float(value) -> Optional[float]:
    try:
        return float(str(value).replace(",", "").rstrip("."))
    except (TypeError, ValueError):
        return None


def answer_correct(text: str, record: dict) -> Optional[int]:
    """1/0 task correctness, or None when the record carries no answer.

    number : the last number in the output must equal the gold value
    yesno  : the first of yes/no appearing in the output must match
    choice : the candidate with the highest unigram F1 against the output must
             be the gold candidate
    """
    kind = record.get("answer_kind")
    gold = record.get("gold")
    if not kind or gold is None:
        return None

    if kind == KIND_NUMBER:
        nums = NUMBER_RE.findall(str(text).replace(",", ""))
        if not nums:
            return 0
        pred, target = _as_float(nums[-1]), _as_float(gold)
        if pred is None or target is None:
            return 0
        return int(abs(pred - target) < 1e-6)

    if kind == KIND_YESNO:
        low = str(text).lower()
        i_yes, i_no = low.find("yes"), low.find("no")
        if i_yes < 0 and i_no < 0:
            return 0
        if i_yes < 0:
            pred = "no"
        elif i_no < 0:
            pred = "yes"
        else:
            pred = "yes" if i_yes < i_no else "no"
        return int(pred == str(gold).strip().lower())

    if kind == KIND_CHOICE:
        candidates: Sequence[str] = record.get("choices") or []
        if not candidates:
            return None
        scores = [unigram_f1(text, c) for c in candidates]
        if not any(math.isfinite(s) and s > 0 for s in scores):
            return 0
        best = max(range(len(scores)), key=lambda i: scores[i])
        return int(candidates[best] == gold)

    return None


# ------------------------------------------------------------ IFEval relations

RELATIONS = ("at least", "less than", "at most", "more than", "exactly")


def ifeval_pass(relation: str, n: int, realized: int) -> bool:
    """Whether a realized word count satisfies a stated relation."""
    if relation == "at least":
        return realized >= n
    if relation == "more than":
        return realized > n
    if relation == "at most":
        return realized <= n
    if relation == "less than":
        return realized < n
    if relation == "exactly":
        return realized == n
    return False


def feasible_distance(relation: str, n: int, realized: int) -> float:
    """Word distance from the realized count to the feasible region."""
    if ifeval_pass(relation, n, realized):
        return 0.0
    if relation in ("at least", "more than"):
        lower = n if relation == "at least" else n + 1
        return float(lower - realized)
    if relation in ("at most", "less than"):
        upper = n if relation == "at most" else n - 1
        return float(realized - upper)
    return float(abs(realized - n))


def scalar_target(relation: str, n: int) -> int:
    """Scalar word target one word inside the feasible region.

    Targeting the boundary itself would mean a one-word error already violates
    the constraint, so the target is placed one word inside.
    """
    if relation == "at least":
        return n + 1
    if relation == "more than":
        return n + 2
    if relation == "at most":
        return max(1, n - 1)
    if relation == "less than":
        return max(1, n - 2)
    return n
