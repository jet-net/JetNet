from jetnet.datasets import JetNet
import numpy as np

import pytest
from pytest import approx


# TODO: use checksum for downloaded files


def test_getData():
    pf, jf = JetNet.getData("g", "./datasets")
    assert pf.shape == (177252, 30, 4)
    assert jf.shape == (177252, 5)
    assert np.all(jf[:, 0] == 0)

    pf, jf = JetNet.getData("w", "./datasets")
    assert pf.shape == (177172, 30, 4)
    assert jf.shape == (177172, 5)
    assert np.all(jf[:, 0] == 3)

    pf, jf = JetNet.getData(data_dir="./datasets")
    assert pf.shape == (880000, 30, 4)
    assert jf.shape == (880000, 5)

    pf, jf = JetNet.getData(data_dir="./datasets/", jet_features=["pt", "num_particles"])
    assert pf.shape == (880000, 30, 4)
    assert jf.shape == (880000, 2)
    assert np.all(jf[0] == approx([1068.62719727, 30.0]))

    pf, jf = JetNet.getData(data_dir="./datasets/", jet_features=None)
    assert pf.shape == (880000, 30, 4)
    assert jf is None

    pf, jf = JetNet.getData(data_dir="./datasets/", particle_features=["etarel", "mask"])
    assert pf.shape == (880000, 30, 2)
    assert jf.shape == (880000, 5)
    assert np.all(pf[0][0] == approx([-0.01513395, 1.0]))

    with pytest.raises(AssertionError):
        JetNet.getData(data_dir="./datasets/", particle_features="foo")

    with pytest.raises(AssertionError):
        JetNet.getData(data_dir="./datasets/", particle_features=["eta", "mask"])

    with pytest.raises(AssertionError):
        JetNet.getData(data_dir="./datasets/", jet_features=["eta", "mask"])


# TODO: JetNet class tests
