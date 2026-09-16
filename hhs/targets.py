"""Target stopping-step distributions and their hazard schedules.

A length request is expressed in words, while the controller acts on stopping
steps. We estimate an empirical tokens-per-word ratio distribution from held-out
human-written text and push a requested word count through it to obtain a
distribution over stopping steps.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

from .textutils import words


def ratio_distribution(tokenizer, texts: Iterable[str]) -> list[float]:
    """Empirical tokens-per-word ratios of the calibration texts.

    Texts with no words are skipped. The ratio uses the tokenizer of the
    checkpoint the controller will run on, since the stopping step is counted in
    that tokenizer's units.
    """
    ratios = []
    for text in texts:
        n_words = len(words(text))
        if n_words <= 0:
            continue
        n_tokens = len(tokenizer.encode(str(text), add_special_tokens=False))
        if n_tokens <= 0:
            continue
        ratios.append(n_tokens / n_words)
    if not ratios:
        ratios = [1.5]
    return ratios


def stopping_step(requested_words: float, ratio: float, max_step: int) -> int:
    """Stopping step implied by one tokens-per-word ratio.

    The stopping token is sampled after the final content token, so an output of
    m content tokens corresponds to stopping at step m + 1. Steps beyond the
    budget are clipped to the budget, which preserves total probability mass.
    """
    step = int(round(requested_words * ratio)) + 1
    return min(max_step, max(1, step))


def make_distribution(requested_words: float,
                      ratios: Sequence[float],
                      point_mass: bool,
                      max_step: int) -> list[float]:
    """Target distribution D over stopping steps 1..max_step.

    With ``point_mass=True`` the whole mass is placed at the step implied by the
    mean ratio, which reproduces the deterministic fixed-ratio controller as the
    degenerate case of the same construction.
    """
    dist = [0.0] * max_step
    if point_mass:
        mean_ratio = sum(ratios) / len(ratios)
        dist[stopping_step(requested_words, mean_ratio, max_step) - 1] = 1.0
        return dist
    for ratio in ratios:
        dist[stopping_step(requested_words, ratio, max_step) - 1] += 1.0
    total = sum(dist)
    return [v / total for v in dist]


def permuted_distribution(distribution: Sequence[float], seed: int) -> list[float]:
    """Rearrange the nonzero masses of D across their original support.

    The permuted target has the same support as D and a different temporal
    allocation of mass. One rearrangement is drawn per condition and reused for
    every example, so the ablation isolates the schedule shape.
    """
    idxs = [i for i, v in enumerate(distribution) if v > 0.0]
    if len(idxs) <= 1:
        return list(distribution)
    vals = [distribution[i] for i in idxs]
    order = list(range(len(vals)))
    random.Random(seed).shuffle(order)
    out = [0.0] * len(distribution)
    for slot, i in enumerate(idxs):
        out[i] = vals[order[slot]]
    return out


def distribution_hazards(distribution: Sequence[float]) -> list[float]:
    """Hazard schedule h*(t) = D(t) / S(t) with S(t) = sum_{u >= t} D(u).

    The final support point has hazard 1, so the induced distribution places no
    mass beyond the budget.
    """
    hazards = [1.0] * len(distribution)
    survival = 0.0
    for i in range(len(distribution) - 1, -1, -1):
        survival += distribution[i]
        hazards[i] = distribution[i] / survival if survival > 0.0 else 1.0
    return hazards


def survival_curve(distribution: Sequence[float]) -> list[float]:
    """S(t) = P(T >= t) for t = 1..len(distribution)."""
    out = []
    tail = float(sum(distribution))
    for value in distribution:
        out.append(max(0.0, tail))
        tail -= value
    return out


def tail_mass(ratios: Sequence[float], requested_words: float, max_step: int) -> float:
    """Fraction of calibration ratios whose implied step exceeds the budget.

    This mass is clipped onto the final feasible step, which raises D(max_step).
    """
    over = sum(1 for r in ratios
               if int(round(requested_words * r)) + 1 > max_step)
    return over / len(ratios)
