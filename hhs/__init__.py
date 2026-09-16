"""Historywise hazard steering.

A training-free decoding intervention that sets the stopping probability of each
generated history to a prescribed target hazard, so that the realized
stopping-time distribution matches an arbitrary finite-support target under
unit-temperature multinomial sampling.

The core of the method is :func:`hhs.controller.steer`.
"""

from .controller import (StepResult, add_constant_bias, force_stop, logit,
                         mask_stop, realized_stop_mass, steer,
                         steer_with_fixed_mass, stop_log_odds)
from .targets import (distribution_hazards, make_distribution,
                      permuted_distribution, ratio_distribution, survival_curve)

__all__ = [
    "StepResult",
    "steer",
    "steer_with_fixed_mass",
    "add_constant_bias",
    "mask_stop",
    "force_stop",
    "stop_log_odds",
    "realized_stop_mass",
    "logit",
    "make_distribution",
    "distribution_hazards",
    "permuted_distribution",
    "ratio_distribution",
    "survival_curve",
]

__version__ = "1.0.0"
