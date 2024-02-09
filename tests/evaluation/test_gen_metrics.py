from __future__ import annotations

import numpy as np
import pytest
from jetnet import evaluation
from pytest import approx

test_zeros = np.zeros((50_000, 2))
test_ones = np.ones((50_000, 2))
test_twos = np.ones((50_000, 2)) * 2


def test_fpd():
    val, err = evaluation.fpd(test_zeros, test_zeros)
    assert val == approx(0, abs=0.01)
    assert err < 1e-3

    val, err = evaluation.fpd(test_twos, test_zeros)
    assert val == approx(2, rel=0.01)  # 1^2 + 1^2
    assert err < 1e-3
    
    # test normalization
    val, err = evaluation.fpd(test_zeros, test_zeros, normalise=False)  # should have no effect
    assert val == approx(0, abs=0.01)
    assert err < 1e-3
    
    val, err = evaluation.fpd(test_twos, test_zeros, normalise=False)
    assert val == approx(8, rel=0.01)  # 2^2 + 2^2
    assert err < 1e-3


@pytest.mark.parametrize("num_threads", [None, 2])  # test numba parallelization
def test_kpd(num_threads):
    assert evaluation.kpd(test_zeros, test_zeros, num_threads=num_threads) == approx([0, 0])
    assert evaluation.kpd(test_twos, test_zeros, num_threads=num_threads) == approx([15, 0])
    
    # test normalization
    assert evaluation.kpd(test_zeros, test_zeros, normalise=False, num_threads=num_threads) == approx([0, 0])
    assert evaluation.kpd(test_twos, test_zeros, normalise=False, num_threads=num_threads) == approx([624, 0])