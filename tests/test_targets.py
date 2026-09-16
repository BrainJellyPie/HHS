"""Target distributions, hazard schedules, and their relationship."""

import pytest

from hhs.targets import (distribution_hazards, make_distribution,
                         permuted_distribution, stopping_step, survival_curve,
                         tail_mass)

RATIOS = [1.1, 1.3, 1.5, 1.7, 1.9, 2.3]
MAX_STEP = 48


def test_distribution_is_normalized():
    dist = make_distribution(24, RATIOS, False, MAX_STEP)
    assert len(dist) == MAX_STEP
    assert abs(sum(dist) - 1.0) < 1e-12
    assert all(v >= 0.0 for v in dist)


def test_hazards_reconstruct_the_distribution():
    """D(t) = S(t) h*(t) with S the survival of the target itself."""
    dist = make_distribution(24, RATIOS, False, MAX_STEP)
    hazards = distribution_hazards(dist)
    survival, recovered = 1.0, []
    for h in hazards:
        recovered.append(survival * h)
        survival *= (1.0 - h)
    assert max(abs(a - b) for a, b in zip(dist, recovered)) < 1e-12


def test_final_support_point_has_unit_hazard():
    """No mass escapes the budget."""
    dist = make_distribution(24, RATIOS, False, MAX_STEP)
    hazards = distribution_hazards(dist)
    last = max(i for i, v in enumerate(dist) if v > 0)
    assert abs(hazards[last] - 1.0) < 1e-12


def test_point_mass_target_is_degenerate():
    dist = make_distribution(24, RATIOS, True, MAX_STEP)
    assert sum(1 for v in dist if v > 0) == 1
    assert abs(sum(dist) - 1.0) < 1e-12


def test_stopping_step_offsets_by_one_for_the_stop_token():
    """m content tokens correspond to stopping at step m + 1."""
    assert stopping_step(10, 1.0, MAX_STEP) == 11
    assert stopping_step(10, 2.0, MAX_STEP) == 21


def test_stopping_step_is_clipped_to_the_budget():
    assert stopping_step(1000, 2.0, MAX_STEP) == MAX_STEP
    assert stopping_step(0, 0.1, MAX_STEP) == 1


def test_clipping_moves_mass_onto_the_final_step():
    dist = make_distribution(60, RATIOS, False, MAX_STEP)
    assert dist[-1] > 0.0
    assert abs(sum(dist) - 1.0) < 1e-12
    assert tail_mass(RATIOS, 60, MAX_STEP) > 0.0


def test_permutation_preserves_support_and_mass():
    dist = make_distribution(24, RATIOS, False, MAX_STEP)
    permuted = permuted_distribution(dist, seed=7)
    assert abs(sum(permuted) - sum(dist)) < 1e-12
    assert {i for i, v in enumerate(dist) if v > 0} == \
           {i for i, v in enumerate(permuted) if v > 0}
    assert sorted(v for v in permuted if v > 0) == sorted(v for v in dist if v > 0)


def test_permutation_is_deterministic_given_the_seed():
    dist = make_distribution(24, RATIOS, False, MAX_STEP)
    assert permuted_distribution(dist, 7) == permuted_distribution(dist, 7)


def test_survival_is_non_increasing_and_starts_at_one():
    dist = make_distribution(24, RATIOS, False, MAX_STEP)
    surv = survival_curve(dist)
    assert abs(surv[0] - 1.0) < 1e-12
    assert all(surv[i] >= surv[i + 1] - 1e-15 for i in range(len(surv) - 1))


@pytest.mark.parametrize("requested", [12, 24, 60])
def test_target_mean_tracks_the_request(requested):
    """Longer requests place their mass at later steps."""
    dist = make_distribution(requested, RATIOS, False, 120)
    mean_step = sum((i + 1) * v for i, v in enumerate(dist))
    assert mean_step > requested
