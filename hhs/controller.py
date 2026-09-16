"""History-conditioned hazard steering.

At every decoding step the controller reads the probability mass that the model
currently places on the designated stopping-token set and adds a closed-form
offset to those logits so that the post-intervention stopping probability equals
a prescribed target hazard for that specific history.

For an interior target hazard ``h`` in (0, 1) the offset is

    b = logit(h) - logit(q)

where ``q`` is the pre-intervention stopping mass. Adding ``b`` to every
stopping-token logit rescales the odds between the stopping set and its
complement by exp(b), so the post-intervention odds are exactly h / (1 - h),
independently of ``q``.

The boundary cases are handled by masking rather than by a diverging offset:
``h = 0`` masks the stopping set, ``h = 1`` masks its complement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

NEG_INF = -float("inf")

MASK_NONE = ""
MASK_ZERO = "zero"   # h = 0, stopping suppressed
MASK_UNIT = "unit"   # h = 1, stopping forced


@dataclass
class StepResult:
    """Outcome of one controller step."""

    logits: torch.Tensor      # modified logits, same shape as the input
    offset: float             # applied offset, nan when a mask was used
    mask: str                 # MASK_NONE / MASK_ZERO / MASK_UNIT
    nonfinite: bool           # True when the numerical guard fired


def logit(p: float) -> float:
    """Log-odds of a probability, numerically safe near 0 and 1."""
    return math.log(p) - math.log1p(-p)


def stop_log_odds(logits: torch.Tensor, stop_ids: Sequence[int]) -> tuple[float, float]:
    """Log-sum-exp over the stopping set and over its complement.

    Both terms share the softmax normalizing constant, so their difference is
    ``log q - log(1 - q)`` without either quantity being formed separately. This
    avoids the precision loss of computing ``log(1 - exp(log q))`` when ``q`` is
    close to zero or one.
    """
    complement = logits.clone()
    complement[:, list(stop_ids)] = NEG_INF
    log_stop = float(torch.logsumexp(logits[:, list(stop_ids)], dim=-1).item())
    log_cont = float(torch.logsumexp(complement, dim=-1).item())
    return log_stop, log_cont


def mask_stop(logits: torch.Tensor, stop_ids: Sequence[int]) -> torch.Tensor:
    """Suppress the stopping set: the post-intervention stopping mass is 0."""
    out = logits.clone()
    out[:, list(stop_ids)] = NEG_INF
    return out


def force_stop(logits: torch.Tensor, stop_ids: Sequence[int]) -> torch.Tensor:
    """Keep only the stopping set: the post-intervention stopping mass is 1.

    The relative preference among stopping tokens is preserved. If the model
    assigns no finite mass to any stopping token, the set is made uniform so
    that the sampling distribution stays well defined.
    """
    ids = list(stop_ids)
    kept = logits[:, ids]
    if not bool(torch.isfinite(kept).any().item()):
        kept = torch.zeros_like(kept)
    out = torch.full_like(logits, NEG_INF)
    out[:, ids] = kept
    return out


def steer(logits: torch.Tensor,
          stop_ids: Sequence[int],
          target_hazard: float) -> StepResult:
    """Apply the history-conditioned offset for one decoding step.

    Args:
        logits: raw next-token logits, shape (1, vocab).
        stop_ids: token ids of the designated stopping set.
        target_hazard: prescribed hazard h*(t) in [0, 1].

    Returns:
        StepResult with the modified logits.
    """
    h = float(target_hazard)

    if h <= 0.0:
        return StepResult(mask_stop(logits, stop_ids), float("nan"), MASK_ZERO, False)
    if h >= 1.0:
        return StepResult(force_stop(logits, stop_ids), float("nan"), MASK_UNIT, False)

    log_stop, log_cont = stop_log_odds(logits, stop_ids)

    if not (math.isfinite(log_stop) and math.isfinite(log_cont)):
        # Numerical guard: fall back to the corresponding boundary mask rather
        # than propagate a non-finite offset into the sampling distribution.
        if h >= 0.5:
            return StepResult(force_stop(logits, stop_ids), float("nan"), MASK_UNIT, True)
        return StepResult(mask_stop(logits, stop_ids), float("nan"), MASK_ZERO, True)

    offset = logit(h) - (log_stop - log_cont)
    out = logits.clone()
    out[:, list(stop_ids)] += offset
    return StepResult(out, float(offset), MASK_NONE, False)


def steer_with_fixed_mass(logits: torch.Tensor,
                          stop_ids: Sequence[int],
                          target_hazard: float,
                          stopping_mass: float) -> StepResult:
    """Ablation variant: use a supplied population stopping mass instead of the
    history-conditioned one.

    This realizes the plug-in marginal-hazard controller analyzed in the paper.
    """
    h = float(target_hazard)
    if h <= 0.0:
        return StepResult(mask_stop(logits, stop_ids), float("nan"), MASK_ZERO, False)
    if h >= 1.0:
        return StepResult(force_stop(logits, stop_ids), float("nan"), MASK_UNIT, False)

    q = min(max(float(stopping_mass), 1e-6), 1.0 - 1e-6)
    offset = logit(h) - logit(q)
    out = logits.clone()
    out[:, list(stop_ids)] += offset
    return StepResult(out, float(offset), MASK_NONE, False)


def add_constant_bias(logits: torch.Tensor,
                      stop_ids: Sequence[int],
                      bias: float) -> torch.Tensor:
    """Baseline: a single shared offset added to the stopping-token logits."""
    out = logits.clone()
    out[:, list(stop_ids)] += float(bias)
    return out


def realized_stop_mass(logits: torch.Tensor,
                       stop_ids: Sequence[int],
                       temperature: float = 1.0) -> float:
    """Probability assigned to the stopping set after temperature scaling.

    The reduction is taken in double precision so that the measurement noise
    stays well below the deviation being measured; the steering itself operates
    on the model's own logits.
    """
    scaled = logits.double() / float(temperature)
    log_stop = torch.logsumexp(scaled[:, list(stop_ids)], dim=-1)
    log_all = torch.logsumexp(scaled, dim=-1)
    return float(torch.exp(log_stop - log_all).item())
