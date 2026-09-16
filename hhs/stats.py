"""Aggregation and inferential statistics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .config import N_BOOT
from .targets import make_distribution, permuted_distribution


def finite_values(rows: Iterable[dict], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def finite_mean(rows: Iterable[dict], key: str) -> float:
    values = finite_values(rows, key)
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: Sequence[float], seed: int = 0,
                 n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap 95% interval over the supplied values."""
    arr = np.array([v for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(arr.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def sign_permutation_p(diffs: Sequence[float], seed: int = 0,
                       n_boot: int = N_BOOT) -> float:
    """Two-sided paired sign-permutation test on per-generation differences.

    The p-value cannot fall below 1 / n_boot, which is the resolution of the
    permutation null.
    """
    d = np.array(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    observed = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
    null = (signs * d).mean(axis=1)
    p = float((np.abs(null) >= observed - 1e-12).mean())
    return min(1.0, max(p, 1.0 / n_boot))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment; non-finite entries are passed through."""
    arr = np.array(pvalues, dtype=float)
    adjusted = np.full(arr.shape, np.nan)
    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return adjusted
    sub = arr[finite_idx]
    order = np.argsort(sub)
    m = sub.size
    running = 0.0
    for rank, oi in enumerate(order):
        running = max(running, (m - rank) * sub[oi])
        adjusted[finite_idx[oi]] = min(1.0, running)
    return adjusted


def wasserstein1(stops: Sequence[int], distribution: Sequence[float]) -> float:
    """W1 between the empirical distribution of stopping steps and a target.

    On a discrete grid this is the sum of absolute CDF differences.
    """
    cdf = np.cumsum(np.array(distribution, dtype=float))
    arr = np.array(stops, dtype=float)
    return float(sum(abs(float((arr <= t).mean()) - float(cdf[t - 1]))
                     for t in range(1, len(distribution) + 1)))


def grouped_wasserstein(rows: Sequence[dict],
                        ratios_by_model: dict,
                        min_stops: int = 5,
                        use_permuted: bool = False,
                        ablation_seed: int = 0) -> float:
    """Average W1 over (model, dataset, requested length, budget) groups.

    Every method is scored against the same target for a given group, so the
    distance measures agreement with a common external reference. Groups with
    fewer than ``min_stops`` observed stopping events are skipped.
    """
    groups = defaultdict(list)
    for row in rows:
        try:
            key = (row["model"], row["dataset"],
                   int(round(float(row["requested_words"]))), int(row["max_step"]))
        except (KeyError, TypeError, ValueError):
            continue
        groups[key].append(row)

    values = []
    for (model, _dataset, target, max_step), group in groups.items():
        ratios = ratios_by_model.get(model)
        if not ratios:
            continue
        stops = [int(r["stop_step"]) for r in group
                 if r.get("event") == 1 and r.get("stop_step")]
        if len(stops) < min_stops:
            continue
        dist = make_distribution(target, ratios, False, max_step)
        if use_permuted:
            dist = permuted_distribution(dist, ablation_seed)
        values.append(wasserstein1(stops, dist))
    return float(np.mean(values)) if values else float("nan")


def aggregate(rows: Sequence[dict]) -> dict:
    """Standard summary of a set of generation rows."""
    return {
        "n": len(rows),
        "mae": finite_mean(rows, "word_error"),
        "compliance": 100.0 * finite_mean(rows, "compliant"),
        "accuracy": 100.0 * finite_mean(rows, "correct"),
        "hazard_error": finite_mean(rows, "hazard_error"),
        "stopset_mass": finite_mean(rows, "stopset_mass"),
        "unigram_f1": finite_mean(rows, "ref_f1"),
        "rouge_l": finite_mean(rows, "rouge_l"),
        "repeat4": finite_mean(rows, "repeat4"),
        "terminal": 100.0 * finite_mean(rows, "terminal"),
        "natural": 100.0 * finite_mean(rows, "natural"),
        "forced": 100.0 * finite_mean(rows, "forced"),
        "external": 100.0 * finite_mean(rows, "external"),
        "budget": 100.0 * finite_mean(rows, "budget"),
        "seconds": finite_mean(rows, "seconds"),
        "words": finite_mean(rows, "words"),
    }
