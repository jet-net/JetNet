from typing import List, Set, Union, Optional, Tuple
from numpy.typing import ArrayLike

import torch
from torch import Tensor
import numpy as np

import logging
from inspect import cleandoc

import os
from os.path import exists

from .dataset import JetDataset
from .utils import (
    download_progress_bar,
    getZenodoFileURL,
    getOrderedFeatures,
    checkStrToList,
    checkListNotEmpty,
    firstNotNoneElement,
    getSplitting,
)
from .normalisations import FeaturewiseLinearBounded, NormaliseABC


class JetNet(JetDataset):
    _zenodo_record_ids = {"30": 6975118, "150": 6302240}

    jet_types = ["g", "t", "q", "w", "z"]
    particle_features_order = ["etarel", "phirel", "ptrel", "mask"]
    jet_features_order = ["type", "pt", "eta", "mass", "num_particles"]
    splits = ["train", "valid", "test", "all"]

    # as used for arXiv:2106.11535
    _default_particle_norm = FeaturewiseLinearBounded(
        feature_norms=1.0, feature_shifts=[0.0, 0.0, -0.5, -0.5]
    )

    # normalisation used for ParticleNet training for FPND, as defined in arXiv:2106.11535
    fpnd_norm = FeaturewiseLinearBounded(
        feature_norms=1.0,
        feature_shifts=[0.0, 0.0, -0.5, 0.0],
        feature_maxes=[1.6211985349655151, 0.520724892616272, 0.8934717178344727, 1.0],
    )

    def __init__(
        self,
        jet_type: Union[str, Set[str]] = "all",
        data_dir: str = "./",
        particle_features: List[str] = particle_features_order,
        jet_features: List[str] = jet_features_order,
        particle_normalisation: NormaliseABC = _default_particle_norm,
        jet_normalisation: NormaliseABC = None,
        num_particles: int = 30,
        split: str = "train",
        split_fraction: List[float] = [0.7, 0.15, 0.15],
        seed: int = 42,
    ):
        """
        PyTorch ``torch.unit.data.Dataset`` class for the JetNet dataset.

        If hdf5 files are not found in the ``data_dir`` directory then dataset will be downloaded
        from Zenodo.

        Args:
            jet_type (Union[str, Set[str]], optional): individual type or set of types out of
                'g' (gluon), 't' (top quarks), 'q' (light quarks), 'w' (W bosons), or 'z' (Z bosons).
                "all" will get all types. Defaults to "all".
            data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
            particle_features (List[str], optional): list of particle features to retrieve. If empty
                or None, gets no particle features. Defaults to
                ``["etarel", "phirel", "ptrel", "mask"]``.
            jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
                gets no particle features. Defaults to
                ``["type", "pt", "eta", "mass", "num_particles"]``.
            particle_normalisation (NormaliseABC, optional): optional normalisation to apply to
                particle data. Defaults to a linear scaling of each feature.
            jet_normalisation (NormaliseABC, optional): optional normalisation to apply to jet data.
                Defaults to None.
            num_particles (int, optional): number of particles to retain per jet, max of 150.
                Defaults to 30.
            split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
                to "train".
            split_fraction (List[float], optional): splitting fraction of training, validation,
                testing data respectively. Defaults to [0.7, 0.15, 0.15].
            seed (int, optional): PyTorch manual seed - important to use the same seed for all
                dataset splittings. Defaults to 42.
        """

        self.particle_data, self.jet_data = JetNet.getData(
            jet_type, data_dir, particle_features, jet_features, num_particles
        )

        super().__init__(
            data_dir=data_dir,
            particle_features=particle_features,
            jet_features=jet_features,
            particle_normalisation=particle_normalisation,
            jet_normalisation=jet_normalisation,
        )

        self.split = split
        self.split_fraction = split_fraction

        # shuffling and splitting into training and test
        lcut, rcut = getSplitting(len(self), self.split, self.splits, self.split_fraction)
        torch.manual_seed(seed)
        randperm = torch.randperm(len(self))

        if self.use_particle_features:
            self.particle_data = self.particle_data[randperm][lcut:rcut]

        if self.use_jet_features:
            self.jet_data = self.jet_data[randperm][lcut:rcut]

    @staticmethod
    def getData(
        jet_type: Union[str, Set[str]] = "all",
        data_dir: str = "./",
        particle_features: List[str] = particle_features_order,
        jet_features: List[str] = jet_features_order,
        num_particles: int = 30,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Downloads, if needed, and loads and returns JetNet data.

        Args:
            jet_type (Union[str, Set[str]], optional): individual type or set of types out of
                'g' (gluon), 't' (top quarks), 'q' (light quarks), 'w' (W bosons), or 'z' (Z bosons).
                "all" will get all types. Defaults to "all".
            data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
            particle_features (List[str], optional): list of particle features to retrieve. If empty
                or None, gets no particle features. Defaults to
                ``["etarel", "phirel", "ptrel", "mask"]``.
            jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
                gets no particle features. Defaults to
                ``["type", "pt", "eta", "mass", "num_particles"]``.
            num_particles (int, optional): number of particles to retain per jet, max of 150.
                Defaults to 30.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: particle data, jet data
        """
        import h5py

        if jet_type != "all":
            jet_type = checkStrToList(jet_type, to_set=True)

            for j in jet_type:
                assert (
                    j in JetNet.jet_types
                ), f"{j} is not a valid jet type, must be one of {JetNet.jet_types}"

        else:
            jet_type = JetNet.jet_types

        particle_features, jet_features = checkStrToList(particle_features, jet_features)
        use_particle_features, use_jet_features = checkListNotEmpty(particle_features, jet_features)

        # Use JetNet150 if ``num_particles`` > 30
        use_150 = num_particles > 30

        particle_data = []
        jet_data = []

        for j in jet_type:
            hdf5_file = JetNet._check_download_dataset(data_dir, j, use_150)

            with h5py.File(hdf5_file, "r") as f:
                pf = (
                    np.array(f["particle_features"])[:, :num_particles]
                    if use_particle_features
                    else None
                )
                jf = np.array(f["jet_features"]) if use_jet_features else None

            if use_particle_features:
                # reorder if needed
                pf = getOrderedFeatures(pf, particle_features, JetNet.particle_features_order)

            if use_jet_features:
                # add class index as first jet feature
                class_index = JetNet.jet_types.index(j)
                jf = np.concatenate((np.full([len(jf), 1], class_index), jf), axis=1)
                # reorder if needed
                jf = getOrderedFeatures(jf, jet_features, JetNet.jet_features_order)

            particle_data.append(pf)
            jet_data.append(jf)

        particle_data = np.concatenate(particle_data, axis=0) if use_particle_features else None
        jet_data = np.concatenate(jet_data, axis=0) if use_jet_features else None

        return particle_data, jet_data

    @staticmethod
    def _check_download_dataset(data_dir: str, jet_type: str, use_150: bool = False) -> str:
        """Checks if dataset exists, if not downloads it from Zenodo, and returns the file path"""
        dname = f"{jet_type}{'150' if use_150 else ''}"
        key = f"{dname}.hdf5"
        hdf5_file = f"{data_dir}/{key}"

        if not exists(hdf5_file):
            os.system(f"mkdir -p {data_dir}")
            record_id = JetNet._zenodo_record_ids["150" if use_150 else "30"]
            file_url = getZenodoFileURL(record_id, key)

            logging.info(f"Downloading {dname} dataset to {hdf5_file}")
            download_progress_bar(file_url, hdf5_file)

        return hdf5_file

    def __getitem__(self, index) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        particle_data_index = self.particle_data[index] if self.use_particle_features else None
        jet_data_index = self.jet_data[index] if self.use_jet_features else None
        return particle_data_index, jet_data_index

    def extra_repr(self) -> str:
        if self.split == "all":
            return ""
        else:
            return (
                f"Split into {self.split} data out of {self.splits} possible splits, "
                f"with splitting fractions {self.split_fraction}"
            )
