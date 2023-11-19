from __future__ import annotations

import numpy as np
import pytest
from jetnet.datasets.normalisations import FeaturewiseLinear, FeaturewiseLinearBounded
from pytest import approx

rng = np.random.generator(42)
test_data_1d = rng.random(3) * 100
test_data_2d = rng.random(4, 3) * 100
test_data_3d = rng.random(5, 4, 3) * 100

test_data_1d_posneg = rng.random(3) * 100 - 50
test_data_2d_posneg = rng.random(4, 3) * 100 - 50
test_data_3d_posneg = rng.random(5, 4, 3) * 100 - 50


@pytest.mark.parametrize(
    "data",
    [
        test_data_1d,
        test_data_2d,
        test_data_3d,
        test_data_1d_posneg,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinearBounded(data):
    norm = FeaturewiseLinearBounded()

    norm.derive_dataset_features(data)
    assert norm.feature_maxes.shape == (data.shape[-1],)
    assert np.all(norm.feature_maxes == np.max(np.abs(data.reshape(-1, 3)), axis=0))

    normed = norm(data)
    assert normed.shape == data.shape
    assert np.all(np.max(np.abs(normed.reshape(-1, 3)), axis=0) == approx(1))

    unnormed = norm(normed, inverse=True)
    assert np.all(unnormed == approx(data))


@pytest.mark.parametrize(
    "data",
    [
        test_data_1d,
        test_data_2d,
        test_data_3d,
        test_data_1d_posneg,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinearBoundedErrors(data):
    norm = FeaturewiseLinearBounded()
    with pytest.raises(AssertionError):
        norm(data)

    norm = FeaturewiseLinearBounded(feature_norms=[3, 5])
    norm.derive_dataset_features(data)
    with pytest.raises(AssertionError):
        norm(data)

    norm = FeaturewiseLinearBounded(feature_shifts=[3, 5])
    norm.derive_dataset_features(data)
    with pytest.raises(AssertionError):
        norm(data)

    norm = FeaturewiseLinearBounded(feature_maxes=[3, 5])
    with pytest.raises(AssertionError):
        norm(data)

    norm = FeaturewiseLinearBounded(normalise_features=[True, False])
    norm.derive_dataset_features(data)
    with pytest.raises(AssertionError):
        norm(data)


@pytest.mark.parametrize(
    "data",
    [
        test_data_1d,
        test_data_2d,
        test_data_3d,
        test_data_1d_posneg,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinearBoundedNones(data):
    norm = FeaturewiseLinearBounded(feature_norms=[None, 8, None], feature_shifts=[3, None, None])
    norm.derive_dataset_features(data)

    normed = norm(data)
    assert normed.shape == data.shape
    assert np.all(normed[..., 0] == approx(data[..., 0] + 3))
    assert np.max(np.abs(normed[..., 1])) == approx(8)
    assert np.all(normed[..., 2] == approx(data[..., 2]))

    unnormed = norm(normed, inverse=True)
    assert np.all(unnormed == approx(data))


@pytest.mark.parametrize(
    "data",
    [
        test_data_1d,
        test_data_2d,
        test_data_3d,
        test_data_1d_posneg,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinearBoundedCustom(data):
    norm = FeaturewiseLinearBounded(
        feature_norms=[3, 8, -1], feature_shifts=[2, 0, 3], normalise_features=[True, False, True]
    )
    norm.derive_dataset_features(data)

    normed = norm(data)
    assert normed.shape == data.shape
    assert np.all(normed[..., 0] == approx(data[..., 0] / np.max(np.abs(data[..., 0])) * 3 + 2))
    assert np.all(normed[..., 1] == normed[..., 1])
    assert np.all(normed[..., 2] == approx(data[..., 2] / np.max(np.abs(data[..., 2])) * (-1) + 3))

    unnormed = norm(normed, inverse=True)
    assert np.all(unnormed == approx(data))


@pytest.mark.parametrize(
    "data",
    [
        test_data_1d,
        test_data_2d,
        test_data_3d,
        test_data_1d_posneg,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinear(data):
    norm = FeaturewiseLinear(
        feature_scales=[5, 0.25, -4],
        feature_shifts=[2, -1, 3],
        normalise_features=[False, True, True],
    )

    norm.derive_dataset_features(data)  # should do nothing

    normed = norm(data)
    assert normed.shape == data.shape
    assert np.all(normed[..., 0] == approx(data[..., 0]))
    assert np.all(normed[..., 1] == approx((data[..., 1] - 1) * 0.25))
    assert np.all(normed[..., 2] == approx((data[..., 2] + 3) * (-4)))

    unnormed = norm(normed, inverse=True)
    assert np.all(unnormed == approx(data))


@pytest.mark.parametrize(
    "data",
    [
        test_data_2d,
        test_data_3d,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinearNormal(data):
    norm = FeaturewiseLinear(normal=True, normalise_features=[True, False, True])
    norm.derive_dataset_features(data)

    normed = norm(data)
    assert normed.shape == data.shape
    assert np.mean(normed[..., 0]) == approx(0)
    assert np.std(normed[..., 0]) == approx(1)
    assert np.mean(normed[..., 2]) == approx(0)
    assert np.std(normed[..., 2]) == approx(1)
    assert np.all(normed[..., 1] == approx(data[..., 1]))

    unnormed = norm(normed, inverse=True)
    assert np.all(unnormed == approx(data))


@pytest.mark.parametrize(
    "data",
    [
        test_data_1d,
        test_data_2d,
        test_data_3d,
        test_data_1d_posneg,
        test_data_2d_posneg,
        test_data_3d_posneg,
    ],
)
def test_FeaturewiseLinearErrors(data):
    norm = FeaturewiseLinear(feature_scales=[3, 5])
    norm.derive_dataset_features(data)
    with pytest.raises(AssertionError):
        norm(data)

    norm = FeaturewiseLinear(feature_shifts=[3, 5])
    norm.derive_dataset_features(data)
    with pytest.raises(AssertionError):
        norm(data)

    norm = FeaturewiseLinear(normalise_features=[False, False])
    with pytest.raises(AssertionError):
        norm(data)
