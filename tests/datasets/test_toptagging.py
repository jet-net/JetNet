from jetnet.datasets import TopTagging, normalisations
import numpy as np

import pytest
from pytest import approx

from torch.utils.data import DataLoader


# TODO: use checksum for downloaded files


data_dir = "./datasets/toptagging"
total_length = 2018000
valid_length = 403000
DataClass = TopTagging


@pytest.mark.parametrize(
    "jet_types,split,expected_length,class_id",
    [
        ("qcd", "all", 1008940, 0),
        ("top", "all", 1009060, 1),
        ("qcd", "valid", 201503, 0),
        ("top", "test", 202086, 1),
        ("top", "train", 605477, 1),
        ("all", "train", 1211000, None),
        ("all", "valid", 403000, None),
        ("all", "test", 404000, None),
        ("all", "all", total_length, None),
    ],
)
@pytest.mark.parametrize("num_particles", [30, 200])
def test_getData(jet_types, split, num_particles, expected_length, class_id):
    pf, jf = DataClass.getData(jet_types, data_dir, num_particles=num_particles, split=split)
    assert pf.shape == (expected_length, num_particles, 4)
    assert jf.shape == (expected_length, 5)
    if class_id is not None:
        assert np.all(jf[:, 0] == class_id)


num_particles = 200


def test_getDataFeatures():
    pf, jf = DataClass.getData(
        data_dir=data_dir, num_particles=num_particles, jet_features=["E", "type"]
    )
    assert pf.shape == (total_length, num_particles, 4)
    assert jf.shape == (total_length, 2)
    assert np.max(jf[:, 0], axis=0) == approx(3000, rel=0.2)
    assert np.max(jf[:, 1], axis=0) == 1

    pf, jf = DataClass.getData(data_dir=data_dir, num_particles=num_particles, jet_features=None)
    assert pf.shape == (total_length, num_particles, 4)
    assert jf is None

    pf, jf = DataClass.getData(
        data_dir=data_dir, num_particles=num_particles, particle_features=["px", "E"]
    )
    assert pf.shape == (total_length, num_particles, 2)
    assert jf.shape == (total_length, 5)
    assert np.max(pf[:, :, 0], axis=0) == approx(700, rel=0.2)
    assert np.max(pf[:, :, 0], axis=0) == approx(2000, rel=0.1)


def test_getDataErrors():
    with pytest.raises(AssertionError):
        DataClass.getData(jet_type="f")

    with pytest.raises(AssertionError):
        DataClass.getData(jet_type={"qcd", "f"})

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, particle_features="foo")

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, jet_features=["eta", "mask"])


# jet_types = ["g", "q"]  # faster testing than using full dataset
# gq_length = 177252 + 170679
split = "valid"


def test_DataClass(num_particles):
    X = DataClass(data_dir=data_dir, num_particles=num_particles, split="all")
    assert len(X) == total_length

    X = DataClass(data_dir=data_dir, num_particles=num_particles, split=split)
    assert len(X) == valid_length

    X_loaded = DataLoader(X)
    pf, jf = next(iter(X_loaded))
    assert pf.shape == (1, num_particles, 4)
    assert jf.shape == (1, 5)

    X = DataClass(
        data_dir=data_dir,
        num_particles=num_particles,
        particle_features=["mask", "ptrel"],
        jet_features=None,
        split=split,
    )
    X_loaded = DataLoader(X)
    pf, jf = next(iter(X_loaded))
    assert pf.shape == (1, num_particles, 2)
    assert jf == []

    X = DataClass(
        data_dir=data_dir, num_particles=num_particles, particle_features=None, split=split
    )
    X_loaded = DataLoader(X)
    pf, jf = next(iter(X_loaded))
    assert pf == []
    assert jf.shape == (1, 5)


def test_DataClassNormalisation(num_particles):
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
