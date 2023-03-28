import numpy as np
import pytest
from pytest import approx

from jetnet import evaluation

test_zeros = np.zeros((50_000, 2))
test_ones = np.ones((50_000, 2))


def test_fpd():
    val, err = evaluation.fpd(test_zeros, test_zeros)
    assert val == approx(0, abs=0.01)
    assert err < 1e-3

    val, err = evaluation.fpd(test_zeros, test_ones)
    assert val == approx(2, rel=0.01)
    assert err < 1e-3


@pytest.mark.parametrize("num_threads", [None, 2])  # test numba parallelization
def test_kpd(num_threads):
    assert evaluation.kpd(test_zeros, test_zeros, num_threads=num_threads) == approx([0, 0])
    assert evaluation.kpd(test_zeros, test_ones, num_threads=num_threads) == approx([15, 0])
