import numpy as np
import pytest

from jetnet.datasets import utils

test_data_2d = np.random.rand(4, 3)
test_data_3d = np.random.rand(5, 4, 3)


@pytest.fixture
def features_order():
    return ["eta", "phi", "pt"]


@pytest.mark.parametrize(
    "data,features,expected",
    [
        (test_data_2d, ["phi", "pt"], test_data_2d[:, 1:]),
        (test_data_2d, ["phi"], test_data_2d[:, 1:2]),
        (test_data_2d, ["pt", "phi"], np.stack((test_data_2d[:, 2], test_data_2d[:, 1]), axis=-1)),
        (test_data_3d, ["phi", "pt"], test_data_3d[:, :, 1:]),
        (test_data_3d, ["phi"], test_data_3d[:, :, 1:2]),
        (
            test_data_3d,
            ["pt", "phi"],
            np.stack((test_data_3d[:, :, 2], test_data_3d[:, :, 1]), axis=-1),
        ),
    ],
)
def test_getOrderedFeatures(data, features, features_order, expected):
    assert np.all(utils.getOrderedFeatures(data, features, features_order) == expected)


@pytest.mark.parametrize(
    "data,features",
    [
        (test_data_2d, ["phi", "pt", "foo"]),
        (test_data_2d, "foo"),
    ],
)
def test_getOrderedFeaturesException(data, features, features_order):
    with pytest.raises(AssertionError):
        utils.getOrderedFeatures(data, features, features_order)


@pytest.mark.parametrize(
    "inputs,to_set,expected",
    [
        (["foo"], False, ["foo"]),
        (["foo"], True, {"foo"}),
        ([["foo"]], False, ["foo"]),
        ([{"foo"}], True, {"foo"}),
        (["foo", ["bar"]], False, [["foo"], ["bar"]]),
        (["foo", ["bar", "boom"]], False, [["foo"], ["bar", "boom"]]),
    ],
)
def test_checkStrToList(inputs, to_set, expected):
    assert utils.checkStrToList(*inputs, to_set=to_set) == expected


@pytest.mark.parametrize(
    "inputs,expected",
    [([[]], False), ([None], False), ([[3]], True), ([[], None, [3]], [False, False, True])],
)
def test_checkListNotEmpty(inputs, expected):
    assert utils.checkListNotEmpty(*inputs) == expected


@pytest.mark.parametrize(
    "inputs,expected",
    [([None, 3], 3), ([None], None), ([3, 5, None], 3)],
)
def test_firstNotNoneElement(inputs, expected):
    assert utils.firstNotNoneElement(*inputs) == expected


tvt_splits = ["train", "valid", "test"]
tvt_splits_all = ["train", "valid", "test", "all"]


@pytest.mark.parametrize(
    "length,split,splits,split_fraction,expected",
    [
        (100, "train", tvt_splits, [0.7, 0.15, 0.15], (0, 70)),
        (100, "valid", tvt_splits, [0.7, 0.15, 0.15], (70, 85)),
        (100, "test", tvt_splits, [0.7, 0.15, 0.15], (85, 100)),
        (100, "train", tvt_splits, [0.5, 0.2, 0.3], (0, 50)),
        (100, "valid", tvt_splits, [0.5, 0.2, 0.3], (50, 70)),
        (100, "test", tvt_splits, [0.5, 0.2, 0.3], (70, 100)),
        (10, "valid", tvt_splits, [0.7, 0.15, 0.15], (7, 8)),
        (10, "test", tvt_splits, [0.7, 0.15, 0.15], (8, 10)),
        (100, "valid", tvt_splits_all, [0.7, 0.15, 0.15], (70, 85)),
        (100, "all", tvt_splits_all, [0.7, 0.15, 0.15], (0, 100)),
        (100, "all", tvt_splits_all, [0.7, 0.15, 0.2], (0, 100)),
    ],
)
def test_getSplitting(length, split, splits, split_fraction, expected):
    assert utils.getSplitting(length, split, splits, split_fraction) == expected
