from jetnet.datasets import JetNet, normalisations
import numpy as np

import pytest
from pytest import approx

from torch.utils.data import DataLoader


# TODO: use checksum for downloaded files


data_dir = "./datasets/jetnet"
total_length = 880000
DataClass = JetNet


@pytest.mark.parametrize(
    "jet_types,expected_length,class_id",
    [
        ("g", 177252, 0),
        ("w", 177172, 3),
        (["q", "z"], 170679 + 176952, None),
        ("all", total_length, None),
    ],
)
@pytest.mark.parametrize("num_particles", [30, 75])
def test_getData(jet_types, num_particles, expected_length, class_id):
    pf, jf = DataClass.getData(jet_types, data_dir, num_particles=num_particles)
    assert pf.shape == (expected_length, num_particles, 4)
    assert jf.shape == (expected_length, 5)
    if class_id is not None:
        assert np.all(jf[:, 0] == class_id)


@pytest.mark.parametrize("num_particles", [30, 75])
def test_getDataFeatures(num_particles):
    pf, jf = DataClass.getData(
        data_dir=data_dir, num_particles=num_particles, jet_features=["pt", "num_particles"]
    )
    assert pf.shape == (total_length, num_particles, 4)
    assert jf.shape == (total_length, 2)
    assert np.max(jf[:, 0], axis=0) == approx(3000, rel=0.1)
    assert np.max(jf[:, 1], axis=0) == num_particles

    pf, jf = DataClass.getData(data_dir=data_dir, num_particles=num_particles, jet_features=None)
    assert pf.shape == (total_length, num_particles, 4)
    assert jf is None

    pf, jf = DataClass.getData(
        data_dir=data_dir, num_particles=num_particles, particle_features=["etarel", "mask"]
    )
    assert pf.shape == (total_length, num_particles, 2)
    assert jf.shape == (total_length, 5)
    assert np.max(pf.reshape(-1, 2), axis=0) == approx([1, 1], rel=1e-3)


jet_types = ["g", "q"]  # faster testing than using full dataset
gq_length = 177252 + 170679


@pytest.mark.parametrize("num_particles", [30, 75])
def test_getDataSplitting(num_particles):
    pf, jf = DataClass.getData(
        jet_type=jet_types,
        data_dir=data_dir,
        num_particles=num_particles,
        split_fraction=[0.6, 0.2, 0.2],
        split="train",
    )
    assert len(pf) == int(gq_length * 0.6)
    assert len(jf) == int(gq_length * 0.6)

    pf, jf = DataClass.getData(
        jet_type=jet_types, data_dir=data_dir, num_particles=num_particles, split="all"
    )
    assert len(pf) == int(gq_length)
    assert len(jf) == int(gq_length)

    pf, jf = DataClass.getData(
        jet_type=jet_types,
        data_dir=data_dir,
        num_particles=num_particles,
        split_fraction=[0.6, 0.2, 0.2],
        split="valid",
    )
    assert len(pf) == int(gq_length * 0.8) - int(gq_length * 0.6)
    assert len(jf) == int(gq_length * 0.8) - int(gq_length * 0.6)

    pf, jf = DataClass.getData(
        jet_type=jet_types,
        data_dir=data_dir,
        num_particles=num_particles,
        split_fraction=[0.5, 0.2, 0.3],
        split="test",
    )
    assert len(pf) == gq_length - int(gq_length * 0.7)
    assert len(jf) == gq_length - int(gq_length * 0.7)


def test_getDataErrors():
    with pytest.raises(AssertionError):
        DataClass.getData(jet_type="f")

    with pytest.raises(AssertionError):
        DataClass.getData(jet_type={"g", "f"})

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, particle_features="foo")

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, jet_features=["eta", "mask"])


@pytest.mark.parametrize("num_particles", [30, 75])
def test_DataClass(num_particles):
    X = DataClass(jet_type=jet_types, data_dir=data_dir, num_particles=num_particles)
    assert len(X) == int(gq_length * 0.7)

    X_loaded = DataLoader(X)
    pf, jf = next(iter(X_loaded))
    assert pf.shape == (1, num_particles, 4)
    assert jf.shape == (1, 5)

    X = DataClass(
        jet_type=jet_types,
        data_dir=data_dir,
        num_particles=num_particles,
        particle_features=["mask", "ptrel"],
        jet_features=None,
    )
    X_loaded = DataLoader(X)
    pf, jf = next(iter(X_loaded))
    assert pf.shape == (1, num_particles, 2)
    assert jf == []

    X = DataClass(
        jet_type=jet_types, data_dir=data_dir, num_particles=num_particles, particle_features=None
    )
    X_loaded = DataLoader(X)
    pf, jf = next(iter(X_loaded))
    assert pf == []
    assert jf.shape == (1, 5)


@pytest.mark.parametrize("num_particles", [30, 75])
def test_DataClassNormalisation(num_particles):
    X = DataClass(
        jet_type=jet_types,
        data_dir=data_dir,
        num_particles=num_particles,
        particle_normalisation=normalisations.FeaturewiseLinearBounded(),
        jet_normalisation=normalisations.FeaturewiseLinearBounded(
            normalise_features=[False, True, True, True, True]
        ),
        split="all",
    )

    assert np.all(np.max(np.abs(X.particle_data.reshape(-1, 4)), axis=0) == approx(1))
    assert np.all(np.max(np.abs(X.jet_data[:, 1:].reshape(-1, 4)), axis=0) == approx(1))
    assert np.max(X.jet_data[:, 0]) == 2
