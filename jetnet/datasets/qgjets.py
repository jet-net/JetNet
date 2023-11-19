from __future__ import annotations

from copy import copy
from typing import Callable

import numpy as np

from .dataset import JetDataset
from .normalisations import NormaliseABC
from .utils import (
    checkConvertElements,
    checkDownloadZenodoDataset,
    checkListNotEmpty,
    checkStrToList,
    getOrderedFeatures,
    getSplitting,
)


class QuarkGluon(JetDataset):
    """
    PyTorch ``torch.unit.data.Dataset`` class for the Quark Gluon Jets dataset. Either jets with
    or without bottom and charm quark jets can be selected (``with_bc`` flag).

    If npz files are not found in the ``data_dir`` directory then dataset will be automatically
    downloaded from Zenodo (https://zenodo.org/record/3164691).

    Args:
        jet_type (Union[str, Set[str]], optional): individual type or set of types out of
            'g' (gluon) and 'q' (light quarks). Defaults to "all".
        data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
        with_bc (bool, optional): with or without bottom and charm quark jets. Defaults to True.
        particle_features (List[str], optional): list of particle features to retrieve. If empty
            or None, gets no particle features. Defaults to
            ``["pt", "eta", "phi", "pdgid"]``.
        jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
            gets no jet features. Defaults to
            ``["type"]``.
        particle_normalisation (NormaliseABC, optional): optional normalisation to apply to
            particle data. Defaults to None.
        jet_normalisation (NormaliseABC, optional): optional normalisation to apply to jet data.
            Defaults to None.
        particle_transform (callable, optional): A function/transform that takes in the particle
            data tensor and transforms it. Defaults to None.
        jet_transform (callable, optional): A function/transform that takes in the jet
            data tensor and transforms it. Defaults to None.
        num_particles (int, optional): number of particles to retain per jet, max of 153.
            Defaults to 153.
        split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
            to "train".
        split_fraction (List[float], optional): splitting fraction of training, validation,
            testing data respectively. Defaults to [0.7, 0.15, 0.15].
        seed (int, optional): PyTorch manual seed - important to use the same seed for all
            dataset splittings. Defaults to 42.
        file_list (List[str], optional): list of files to load, if full dataset is not required.
            Defaults to None (will load all files).
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in the ``data_dir`` directory. If dataset is already downloaded, it is not
            downloaded again. Defaults to False.
    """

    _ZENODO_RECORD_ID = 3164691

    # False - without bc, True - with bc
    _FILE_LIST = {
        False: [
            "QG_jets.npz",
            "QG_jets_1.npz",
            "QG_jets_2.npz",
            "QG_jets_3.npz",
            "QG_jets_4.npz",
            "QG_jets_5.npz",
            "QG_jets_6.npz",
            "QG_jets_7.npz",
            "QG_jets_8.npz",
            "QG_jets_9.npz",
            "QG_jets_10.npz",
            "QG_jets_11.npz",
            "QG_jets_12.npz",
            "QG_jets_13.npz",
            "QG_jets_14.npz",
            "QG_jets_15.npz",
            "QG_jets_16.npz",
            "QG_jets_17.npz",
            "QG_jets_18.npz",
            "QG_jets_19.npz",
        ],
        True: [
            "QG_jets_withbc_0.npz",
            "QG_jets_withbc_1.npz",
            "QG_jets_withbc_2.npz",
            "QG_jets_withbc_3.npz",
            "QG_jets_withbc_3.npz",
            "QG_jets_withbc_4.npz",
            "QG_jets_withbc_5.npz",
            "QG_jets_withbc_6.npz",
            "QG_jets_withbc_7.npz",
            "QG_jets_withbc_8.npz",
            "QG_jets_withbc_9.npz",
            "QG_jets_withbc_10.npz",
            "QG_jets_withbc_11.npz",
            "QG_jets_withbc_12.npz",
            "QG_jets_withbc_13.npz",
            "QG_jets_withbc_14.npz",
            "QG_jets_withbc_15.npz",
            "QG_jets_withbc_16.npz",
            "QG_jets_withbc_17.npz",
            "QG_jets_withbc_18.npz",
            "QG_jets_withbc_19.npz",
        ],
    }

    MAX_NUM_PARTICLES = 153

    JET_TYPES = ["g", "q"]
    ALL_PARTICLE_FEATURES = ["pt", "eta", "phi", "pdgid"]
    ALL_JET_FEATURES = ["type"]
    SPLITS = ["train", "valid", "test", "all"]

    def __init__(
        self,
        jet_type: str | set[str] = "all",
        data_dir: str = "./",
        with_bc: bool = True,
        particle_features: list[str] | None = "all",
        jet_features: list[str] | None = "all",
        particle_normalisation: NormaliseABC | None = None,
        jet_normalisation: NormaliseABC | None = None,
        particle_transform: Callable | None = None,
        jet_transform: Callable | None = None,
        num_particles: int = MAX_NUM_PARTICLES,
        split: str = "train",
        split_fraction: list[float] | None = None,
        seed: int = 42,
        file_list: list[str] | None = None,
        download: bool = False,
    ):
        if particle_features == "all":
            particle_features = copy(self.ALL_PARTICLE_FEATURES)

        if jet_features == "all":
            jet_features = copy(self.ALL_JET_FEATURES)

        if split_fraction is None:
            split_fraction = [0.7, 0.15, 0.15]

        self.particle_data, self.jet_data = self.getData(
            jet_type,
            data_dir,
            with_bc,
            particle_features,
            jet_features,
            num_particles,
            split,
            split_fraction,
            seed,
            file_list,
            download,
        )

        super().__init__(
            data_dir=data_dir,
            particle_features=particle_features,
            jet_features=jet_features,
            particle_normalisation=particle_normalisation,
            jet_normalisation=jet_normalisation,
            particle_transform=particle_transform,
            jet_transform=jet_transform,
            num_particles=num_particles,
        )

        self.jet_type = jet_type
        self.split = split
        self.split_fraction = split_fraction

    @classmethod
    def getData(
        cls: JetDataset,
        jet_type: str | set[str] = "all",
        data_dir: str = "./",
        with_bc: bool = True,
        particle_features: list[str] | None = "all",
        jet_features: list[str] | None = "all",
        num_particles: int = MAX_NUM_PARTICLES,
        split: str = "all",
        split_fraction: list[float] | None = None,
        seed: int = 42,
        file_list: list[str] | None = None,
        download: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Downloads, if needed, and loads and returns Quark Gluon data.

        Args:
            jet_type (Union[str, Set[str]], optional): individual type or set of types out of
                'g' (gluon) and 'q' (light quarks). Defaults to "all".
            data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
            with_bc (bool, optional): with or without bottom and charm quark jets. Defaults to True.
            particle_features (List[str], optional): list of particle features to retrieve. If empty
                or None, gets no particle features. Defaults to
                ``["pt", "eta", "phi", "pdgid"]``.
            jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
                gets no jet features. Defaults to
                ``["type"]``.
            num_particles (int, optional): number of particles to retain per jet, max of 153.
                Defaults to 153.
            split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
                to "train".
            split_fraction (List[float], optional): splitting fraction of training, validation,
                testing data respectively. Defaults to [0.7, 0.15, 0.15].
            seed (int, optional): PyTorch manual seed - important to use the same seed for all
                dataset splittings. Defaults to 42.
            file_list (List[str], optional): list of files to load, if full dataset is not required.
                Defaults to None (will load all files).
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in the ``data_dir`` directory. If dataset is already downloaded, it is not
                downloaded again. Defaults to False.

        Returns:
            tuple[np.ndarray | None, np.ndarray | None]: particle data, jet data
        """
        if particle_features == "all":
            particle_features = copy(cls.ALL_PARTICLE_FEATURES)

        if jet_features == "all":
            jet_features = copy(cls.ALL_JET_FEATURES)

        if split_fraction is None:
            split_fraction = [0.7, 0.15, 0.15]

        assert num_particles <= cls.MAX_NUM_PARTICLES, (
            f"num_particles {num_particles} exceeds max number of "
            + f"particles in the dataset {cls.MAX_NUM_PARTICLES}"
        )

        jet_type = checkConvertElements(jet_type, cls.JET_TYPES, ntype="jet type")
        type_indices = [cls.JET_TYPES.index(t) for t in jet_type]

        particle_features, jet_features = checkStrToList(particle_features, jet_features)
        use_particle_features, use_jet_features = checkListNotEmpty(particle_features, jet_features)

        particle_data = []
        jet_data = []

        file_list = cls._FILE_LIST[with_bc] if file_list is None else file_list

        for file_name in file_list:
            npz_file = checkDownloadZenodoDataset(
                data_dir,
                dataset_name=file_name,
                record_id=cls._ZENODO_RECORD_ID,
                key=file_name,
                download=download,
            )

            print(f"Loading {file_name}")
            data = np.load(npz_file)

            # select only specified types of jets (qcd or top or both)
            jet_selector = np.sum([data["y"] == i for i in type_indices], axis=0).astype(bool)

            if use_particle_features:
                pf = data["X"][jet_selector][:, :num_particles]

                # zero-pad if needed (datasets have different numbers of max particles)
                pf_np = pf.shape[1]
                if pf_np < num_particles:
                    pf = np.pad(pf, ((0, 0), (0, num_particles - pf_np), (0, 0)), constant_values=0)

                # reorder if needed
                pf = getOrderedFeatures(pf, particle_features, cls.ALL_PARTICLE_FEATURES)

            if use_jet_features:
                jf = data["y"][jet_selector].reshape(-1, 1)
                jf = getOrderedFeatures(jf, jet_features, cls.ALL_JET_FEATURES)

            length = np.sum(jet_selector)

            # shuffling and splitting into training and test
            lcut, rcut = getSplitting(length, split, cls.SPLITS, split_fraction)

            rng = np.random.default_rng(seed)
            randperm = rng.permutation(length)

            if use_particle_features:
                pf = pf[randperm][lcut:rcut]
                particle_data.append(pf)

            if use_jet_features:
                jf = jf[randperm][lcut:rcut]
                jet_data.append(jf)

        particle_data = np.concatenate(particle_data, axis=0) if use_particle_features else None
        jet_data = np.concatenate(jet_data, axis=0) if use_jet_features else None

        return particle_data, jet_data

    def extra_repr(self) -> str:
        ret = f"Including {self.jet_type} jets"

        if self.split == "all":
            ret += "\nUsing all data (no split)"
        else:
            ret += (
                f"\nSplit into {self.split} data out of {self.SPLITS} possible splits, "
                f"with splitting fractions {self.split_fraction}"
            )

        return ret
