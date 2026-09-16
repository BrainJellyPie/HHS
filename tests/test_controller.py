"""The per-step identity and its boundary cases."""

import math

import pytest
import torch

from hhs import controller as ctrl

VOCAB = 256
STOP_IDS = [5, 9]


def random_logits(seed: int, scale: float = 5.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, VOCAB, generator=generator) * scale


@pytest.mark.parametrize("hazard", [1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 1 - 1e-4])
def test_interior_hazard_is_realized_exactly(hazard):
    """The post-intervention stopping mass equals the target hazard."""
    for seed in range(20):
        logits = random_logits(seed)
        result = ctrl.steer(logits, STOP_IDS, hazard)
        realized = ctrl.realized_stop_mass(result.logits, STOP_IDS)
        assert abs(realized - hazard) < 1e-6
        assert result.mask == ctrl.MASK_NONE
        assert math.isfinite(result.offset)


def test_identity_holds_regardless_of_pre_intervention_mass():
    """Histories with very different stopping mass reach the same hazard."""
    target = 0.3
    realized = []
    for shift in (-20.0, -5.0, 0.0, 5.0, 20.0):
        logits = random_logits(1)
        logits[:, STOP_IDS] += shift
        result = ctrl.steer(logits, STOP_IDS, target)
        realized.append(ctrl.realized_stop_mass(result.logits, STOP_IDS))
    assert max(abs(v - target) for v in realized) < 1e-6


def test_offset_matches_closed_form():
    """The applied offset equals logit(h) - logit(q)."""
    logits = random_logits(3)
    before = ctrl.realized_stop_mass(logits, STOP_IDS)
    target = 0.42
    result = ctrl.steer(logits, STOP_IDS, target)
    expected = ctrl.logit(target) - ctrl.logit(before)
    assert abs(result.offset - expected) < 1e-6


def test_zero_hazard_suppresses_stopping():
    logits = random_logits(4)
    result = ctrl.steer(logits, STOP_IDS, 0.0)
    assert result.mask == ctrl.MASK_ZERO
    assert ctrl.realized_stop_mass(result.logits, STOP_IDS) == 0.0


def test_unit_hazard_forces_stopping():
    logits = random_logits(5)
    result = ctrl.steer(logits, STOP_IDS, 1.0)
    assert result.mask == ctrl.MASK_UNIT
    assert abs(ctrl.realized_stop_mass(result.logits, STOP_IDS) - 1.0) < 1e-9


def test_forced_stop_preserves_relative_preference():
    """Masking the complement keeps the ordering among stopping tokens."""
    logits = random_logits(6)
    result = ctrl.steer(logits, STOP_IDS, 1.0)
    before = logits[0, STOP_IDS]
    after = result.logits[0, STOP_IDS]
    assert torch.allclose(before - before.max(), after - after.max(), atol=1e-6)


def test_non_stop_logits_are_untouched_for_interior_hazards():
    logits = random_logits(7)
    result = ctrl.steer(logits, STOP_IDS, 0.6)
    keep = [i for i in range(VOCAB) if i not in STOP_IDS]
    assert torch.allclose(logits[0, keep], result.logits[0, keep])


def test_guard_fires_on_non_finite_logits():
    """A degenerate stopping set falls back to a mask instead of a nan offset."""
    logits = random_logits(8)
    logits[:, STOP_IDS] = -float("inf")
    result = ctrl.steer(logits, STOP_IDS, 0.7)
    assert result.nonfinite
    assert result.mask in (ctrl.MASK_ZERO, ctrl.MASK_UNIT)
    assert not math.isnan(ctrl.realized_stop_mass(result.logits, STOP_IDS))


def test_marginal_variant_ignores_history():
    """The plug-in ablation reaches the target only at the supplied mass."""
    target, supplied = 0.3, 0.05
    logits = random_logits(9)
    actual = ctrl.realized_stop_mass(logits, STOP_IDS)
    result = ctrl.steer_with_fixed_mass(logits, STOP_IDS, target, supplied)
    realized = ctrl.realized_stop_mass(result.logits, STOP_IDS)
    assert abs(realized - target) > 1e-6 or abs(actual - supplied) < 1e-9


def test_constant_bias_does_not_reach_a_fixed_target():
    """One shared offset cannot equalize histories with different masses."""
    bias = 3.0
    realized = []
    for shift in (-10.0, 0.0, 10.0):
        logits = random_logits(10)
        logits[:, STOP_IDS] += shift
        realized.append(ctrl.realized_stop_mass(
            ctrl.add_constant_bias(logits, STOP_IDS, bias), STOP_IDS))
    assert max(realized) - min(realized) > 0.1
