from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jetnet.datasets import TopTagging, normalisations
from pytest import approx

# TODO: use checksum for downloaded files


data_dir = Path("./datasets/toptagging")
DataClass = TopTagging

valid_length = 403000
num_particles = 200
split = "valid"  # for faster testing


@pytest.mark.slow
@pytest.mark.parametrize(
    ("jet_types", "split", "expected_length", "class_id"),
    [
        # ("qcd", "all", 1008940, 0),
        # ("top", "all", 1009060, 1),
        ("qcd", "valid", 201503, 0),
        ("top", "valid", 201497, 1),
        # ("top", "test", 202086, 1),
        # ("top", "train", 605477, 1),
        # ("all", "train", 1211000, None),
        ("all", "valid", valid_length, None),
        # ("all", "test", 404000, None),
        # ("all", "all", total_length, None),
    ],
)
def test_getData(jet_types, split, expected_length, class_id):
    # test md5 checksum is working for one of the datasets
    if jet_types == "top" and split == "valid":
        file_path = data_dir / "val.h5"

        if file_path.is_file():
            file_path.unlink()

        # should raise a RunetimeError since file doesn't exist
        with pytest.raises(RuntimeError):
            DataClass.getData(jet_types, data_dir, split=split)

        # write random data to file
        with file_path.open("wb") as f:
            f.write(np.random.bytes(100))  # noqa: NPY002

        # should raise a RunetimeError since file exists but is incorrect
        with pytest.raises(RuntimeError):
            DataClass.getData(jet_types, data_dir, split=split)

    pf, jf = DataClass.getData(jet_types, data_dir, split=split, download=True)
    assert pf.shape == (expected_length, num_particles, 4)
    assert jf.shape == (expected_length, 5)
    if class_id is not None:
        assert np.all(jf[:, 0] == class_id)


@pytest.mark.slow
def test_getDataFeatures():
    pf, jf = DataClass.getData(data_dir=data_dir, jet_features=["E", "type"], split=split)
    assert pf.shape == (valid_length, num_particles, 4)
    assert jf.shape == (valid_length, 2)
    assert np.max(jf[:, 0]) == approx(4000, rel=0.2)
    assert np.max(jf[:, 1]) == 1

    pf, jf = DataClass.getData(data_dir=data_dir, jet_features=None, split=split)
    assert pf.shape == (valid_length, num_particles, 4)
    assert jf is None

    pf, jf = DataClass.getData(
        data_dir=data_dir, particle_features=["px", "E"], num_particles=30, split=split
    )
    assert pf.shape == (valid_length, 30, 2)
    assert jf.shape == (valid_length, 5)
    assert np.max(pf[:, :, 0]) == approx(700, rel=0.2)
    assert np.max(pf[:, :, 1]) == approx(2000, rel=0.2)


@pytest.mark.slow
def test_DataClassNormalisation():
    X = DataClass(
        data_dir=data_dir,
        num_particles=num_particles,
        particle_normalisation=normalisations.FeaturewiseLinearBounded(),
        jet_normalisation=normalisations.FeaturewiseLinearBounded(
            normalise_features=[False, True, True, True, True]
        ),
        split=split,
    )

    assert np.all(np.max(np.abs(X.particle_data.reshape(-1, 4)), axis=0) == approx(1))
    assert np.all(np.max(np.abs(X.jet_data[:, 1:].reshape(-1, 4)), axis=0) == approx(1))
    assert np.max(X.jet_data[:, 0]) == 1


@pytest.mark.slow
def test_getDataErrors():
    with pytest.raises(AssertionError):
        DataClass.getData(jet_type="f", split=split)

    with pytest.raises(AssertionError):
        DataClass.getData(jet_type={"qcd", "f"}, split=split)

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, particle_features="foo", split=split)

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, jet_features=["eta", "mask"], split=split)
