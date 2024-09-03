from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from jetnet.datasets import QuarkGluon
from pytest import approx

# TODO: use checksum for downloaded files


test_file_list_withbc = [
    "QG_jets_withbc_0.npz",
    "QG_jets_withbc_1.npz",
]

test_file_list_withoutbc = [
    "QG_jets.npz",
    "QG_jets_1.npz",
]

data_dir = Path("./datasets/qgjets")
total_length = 200_000
DataClass = QuarkGluon
num_particles = 153


@pytest.mark.slow
@pytest.mark.parametrize(
    ("jet_types", "split", "expected_length", "class_id"),
    [
        ("g", "all", total_length / 2, 0),
        ("q", "train", total_length * 0.7 / 2, 1),
        ("all", "valid", total_length * 0.15, None),
    ],
)
@pytest.mark.parametrize("file_list", [test_file_list_withbc, test_file_list_withoutbc])
def test_getData(jet_types, split, expected_length, class_id, file_list):
    # test md5 checksum is working for one of the datasets
    if jet_types == "q" and file_list == test_file_list_withoutbc:
        file_path = data_dir / file_list[-1]

        if file_path.is_file():
            file_path.unlink()

        # should raise a RunetimeError since file doesn't exist
        with pytest.raises(RuntimeError):
            DataClass.getData(jet_types, data_dir, file_list=file_list, split=split)

        # write random data to file
        with file_path.open("wb") as f:
            f.write(np.random.bytes(100))  # noqa: NPY002

        # should raise a RunetimeError since file exists but is incorrect
        with pytest.raises(RuntimeError):
            DataClass.getData(jet_types, data_dir, file_list=file_list, split=split)

    pf, jf = DataClass.getData(jet_types, data_dir, file_list=file_list, split=split, download=True)
    assert pf.shape == (expected_length, num_particles, 4)
    assert jf.shape == (expected_length, 1)
    if class_id is not None:
        assert np.all(jf[:, 0] == class_id)


@pytest.mark.slow
@pytest.mark.parametrize("file_list", [test_file_list_withbc, test_file_list_withoutbc])
def test_getDataFeatures(file_list):
    pf, jf = DataClass.getData(data_dir=data_dir, jet_features=None, file_list=file_list)
    assert pf.shape == (total_length, num_particles, 4)
    assert jf is None

    pf, jf = DataClass.getData(
        data_dir=data_dir,
        particle_features=["pdgid", "pt"],
        num_particles=30,
        file_list=file_list,
    )
    assert pf.shape == (total_length, 30, 2)
    assert jf.shape == (total_length, 1)
    assert np.max(pf[:, :, 0]) == approx(2212)
    assert np.max(pf[:, :, 1]) == approx(550, rel=0.2)


@pytest.mark.slow
def test_getDataErrors():
    with pytest.raises(AssertionError):
        DataClass.getData(jet_type="f")

    with pytest.raises(AssertionError):
        DataClass.getData(jet_type={"qcd", "f"})

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, particle_features="foo")

    with pytest.raises(AssertionError):
        DataClass.getData(data_dir=data_dir, jet_features=["eta", "mask"])
