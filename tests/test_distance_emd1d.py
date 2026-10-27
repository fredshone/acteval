import numpy as np
import pytest
from ot import emd2_1d

from acteval.distance.wasserstein import emd1d


def _pot(a, b):
    (ak, aw), (bk, bw) = a, b
    return emd2_1d(ak, bk, aw / aw.sum(), bw / bw.sum(), metric="cityblock")


@pytest.mark.parametrize("seed", range(20))
def test_emd1d_matches_pot(seed):
    rng = np.random.default_rng(seed)
    a = (rng.normal(size=rng.integers(1, 50)), None)
    b = (rng.normal(1, 2, size=rng.integers(1, 50)), None)
    a = (a[0], rng.integers(1, 10, size=len(a[0])).astype(float))
    b = (b[0], rng.integers(1, 10, size=len(b[0])).astype(float))
    assert emd1d(a, b) == pytest.approx(_pot(a, b), rel=1e-9, abs=1e-12)


def test_emd1d_shared_and_unsorted_support():
    a = (np.array([3.0, 1.0, 2.0]), np.array([1, 2, 3]))
    b = (np.array([2.0, 3.0, 0.0]), np.array([5, 1, 1]))
    assert emd1d(a, b) == pytest.approx(_pot(a, b), rel=1e-9)


def test_emd1d_identical_is_zero():
    a = (np.array([0.1, 0.5]), np.array([3, 7]))
    assert emd1d(a, a) == 0.0


def test_emd1d_zero_weight_side_treated_as_point_mass_at_zero():
    a = (np.array([2.0]), np.array([0]))
    b = (np.array([1.0, 3.0]), np.array([1, 1]))
    assert emd1d(a, b) == pytest.approx(2.0)
