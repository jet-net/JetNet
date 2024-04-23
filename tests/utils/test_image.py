from __future__ import annotations

import numpy as np
import pytest
from jetnet.utils import to_image

# 1 jet with 3 test particles
test_data_2d = np.zeros((3, 3))
test_data_2d[:, 0] = np.array([-0.5, 0, 0.5])  # eta
test_data_2d[:, 1] = np.array([-0.5, 0, 0.5])  # phi
test_data_2d[:, 2] = np.array([1, 1, 1])  # pt
expected_2d = np.identity(3)

# 2 jets
test_data_3d = np.stack([test_data_2d] * 2)
expected_3d = np.stack([expected_2d] * 2)


@pytest.mark.parametrize(
    ("data", "expected"), [(test_data_2d, expected_2d), (test_data_3d, expected_3d)]
)
def test_to_image(data, expected):
    jet_image = to_image(data, im_size=3, maxR=1.0)
    assert len(jet_image.shape) == len(data.shape), "wrong jet image shape"
    assert jet_image.shape[-2:] == (3, 3), "wrong jet image size"
    np.testing.assert_allclose(jet_image, expected)
