"""Numerical sanity checks of the scoring-rule theory shown in the talk.

All scores use the reward orientation of the slides: higher is better, and a
proper score satisfies S(Q, Q) >= S(P, Q), i.e. the expected score is
maximized by reporting the true distribution.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the figure code importable without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

from visualize_theory import crps_normal  # noqa: E402


# --- expected scores for a binary event with true probability q -------------

def expected_brier(p: np.ndarray, q: float) -> np.ndarray:
    """E_Q[Brier reward] = -(q (1-p)^2 + (1-q) p^2)."""
    return -(q * (1 - p) ** 2 + (1 - q) * p**2)


def expected_log(p: np.ndarray, q: float) -> np.ndarray:
    """E_Q[log score] = q log p + (1-q) log(1-p)."""
    return q * np.log(p) + (1 - q) * np.log(1 - p)


def expected_zero_one(p: np.ndarray, q: float) -> np.ndarray:
    """E_Q[zero-one score]: reward q if p > 1/2, 1-q if p < 1/2."""
    return np.where(p > 0.5, q, np.where(p < 0.5, 1 - q, 0.5))


# Interior grid of reported probabilities; avoids log(0) at p = 0 and p = 1.
P_GRID = np.linspace(0.001, 0.999, 1999)


# --- properness -------------------------------------------------------------

@pytest.mark.parametrize("q", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_brier_expected_score_is_maximized_at_truth(q):
    scores = expected_brier(P_GRID, q)
    assert np.all(expected_brier(np.array(q), q) >= scores - 1e-12)
    # and the optimum is unique: the grid argmax sits at the truth
    assert abs(P_GRID[np.argmax(scores)] - q) < 1e-3


@pytest.mark.parametrize("q", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_log_expected_score_is_maximized_at_truth(q):
    scores = expected_log(P_GRID, q)
    assert np.all(expected_log(np.array(q), q) >= scores - 1e-12)
    assert abs(P_GRID[np.argmax(scores)] - q) < 1e-3


def test_zero_one_score_is_not_strictly_proper():
    """Two different reports achieve the same optimal expected score."""
    q = 0.7
    optimum = expected_zero_one(np.array(q), q)
    for p_other in [0.6, 0.9, 0.51]:  # all on the correct side of 1/2
        assert expected_zero_one(np.array(p_other), q) == pytest.approx(optimum)


# --- divergence d(P, Q) = S(Q, Q) - S(P, Q) ---------------------------------

@pytest.mark.parametrize("expected_score", [expected_brier, expected_log])
def test_divergence_is_nonnegative(expected_score):
    q_grid = np.linspace(0.05, 0.95, 19)
    for q in q_grid:
        d = expected_score(np.array(q), q) - expected_score(P_GRID, q)
        assert np.all(d >= -1e-12)


@pytest.mark.parametrize("expected_score", [expected_brier, expected_log])
def test_divergence_is_zero_at_truth(expected_score):
    for q in [0.2, 0.5, 0.8]:
        d = expected_score(np.array(q), q) - expected_score(np.array(q), q)
        assert d == pytest.approx(0.0, abs=1e-14)


# --- CRPS -------------------------------------------------------------------

def test_crps_of_standard_normal_at_center_matches_closed_form():
    """CRPS(N(0,1), 0) = 2 phi(0) - 1/sqrt(pi) = (sqrt(2)-1)/sqrt(pi)."""
    expected = (np.sqrt(2) - 1) / np.sqrt(np.pi)
    assert crps_normal(0.0, 1.0, np.array(0.0)) == pytest.approx(expected)


def test_crps_scales_linearly_with_sigma_at_center():
    base = crps_normal(0.0, 1.0, np.array(0.0))
    assert crps_normal(0.0, 3.0, np.array(0.0)) == pytest.approx(3 * base)


def test_crps_approaches_absolute_error_for_point_forecast():
    """As sigma -> 0 the CRPS of N(mu, sigma^2) tends to |x - mu|."""
    x = np.array([-2.0, -0.5, 1.0, 3.0])
    crps = crps_normal(0.0, 1e-9, x)
    assert np.allclose(crps, np.abs(x), atol=1e-6)


def test_crps_is_nonnegative_and_grows_with_distance():
    obs = np.linspace(0.0, 5.0, 50)
    crps = crps_normal(0.0, 1.0, obs)
    assert np.all(crps >= 0)          # a loss in positive orientation
    assert np.all(np.diff(crps) > 0)  # farther misses cost more
