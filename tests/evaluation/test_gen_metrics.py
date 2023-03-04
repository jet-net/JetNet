import numpy as np
import pytest
from pytest import approx

from jetnet import evaluation

test_zeros = np.zeros((50_000, 2))
test_ones = np.ones((50_000, 2))


def test_fpd():
    val, err = evaluation.fpd_inf(test_zeros, test_zeros)
    assert val == approx(0, abs=0.01)
    assert err < 1e-3

    val, err = evaluation.fpd_inf(test_zeros, test_ones)
    assert val == approx(2, rel=0.01)
    assert err < 1e-3


def test_kpd():
    assert evaluation.kpd(test_zeros, test_zeros) == approx([0, 0])
    assert evaluation.kpd(test_zeros, test_ones) == approx([15, 0])
