from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np

from .dataset import JetDataset
from .normalisations import FeaturewiseLinearBounded, NormaliseABC
from .utils import (
    checkConvertElements,
    checkDownloadZenodoDataset,
    checkListNotEmpty,
    checkStrToList,
    firstNotNoneElement,
    getOrderedFeatures,
    getSplitting,
)


class JetNet(JetDataset):
    """
    PyTorch ``torch.unit.data.Dataset`` class for the JetNet dataset.

    If hdf5 files are not found in the ``data_dir`` directory then dataset will be downloaded
    from Zenodo (https://zenodo.org/record/6975118 or https://zenodo.org/record/6975117).

    Args:
        jet_type (Union[str, Set[str]], optional): individual type or set of types out of
            'g' (gluon), 'q' (light quarks), 't' (top quarks), 'w' (W bosons), or 'z' (Z bosons).
            "all" will get all types. Defaults to "all".
        data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
        particle_features (List[str], optional): list of particle features to retrieve. If empty
            or None, gets no particle features. Defaults to
            ``["etarel", "phirel", "ptrel", "mask"]``.
        jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
            gets no jet features. Defaults to
            ``["type", "pt", "eta", "mass", "num_particles"]``.
        particle_normalisation (NormaliseABC, optional): optional normalisation to apply to
            particle data. Defaults to None.
        jet_normalisation (NormaliseABC, optional): optional normalisation to apply to jet data.
            Defaults to None.
        particle_transform (callable, optional): A function/transform that takes in the particle
            data tensor and transforms it. Defaults to None.
        jet_transform (callable, optional): A function/transform that takes in the jet
            data tensor and transforms it. Defaults to None.
        num_particles (int, optional): number of particles to retain per jet, max of 150.
            Defaults to 30.
        split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
            to "train".
        split_fraction (List[float], optional): splitting fraction of training, validation,
            testing data respectively. Defaults to [0.7, 0.15, 0.15].
        seed (int, optional): PyTorch manual seed - important to use the same seed for all
            dataset splittings. Defaults to 42.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in the ``data_dir`` directory. If dataset is already downloaded, it is not
            downloaded again. Defaults to False.
    """

    _zenodo_record_ids = {"30": 6975118, "150": 6975117}

    max_num_particles = 150

    jet_types = ["g", "q", "t", "w", "z"]
    all_particle_features = ["etarel", "phirel", "ptrel", "mask"]
    all_jet_features = ["type", "pt", "eta", "mass", "num_particles"]
    splits = ["train", "valid", "test", "all"]

    # normalisation used for ParticleNet training for FPND, as defined in arXiv:2106.11535
    fpnd_norm = FeaturewiseLinearBounded(
        feature_norms=1.0,
        feature_shifts=[0.0, 0.0, -0.5],
        feature_maxes=[1.6211985349655151, 0.520724892616272, 0.8934717178344727],
    )

    def __init__(
        self,
        jet_type: Union[str, Set[str]] = "all",
        data_dir: str = "./",
        particle_features: List[str] = all_particle_features,
        jet_features: List[str] = all_jet_features,
        particle_normalisation: Optional[NormaliseABC] = None,
        jet_normalisation: Optional[NormaliseABC] = None,
        particle_transform: Optional[Callable] = None,
        jet_transform: Optional[Callable] = None,
        num_particles: int = 30,
        split: str = "train",
        split_fraction: List[float] = [0.7, 0.15, 0.15],
        seed: int = 42,
        download: bool = False,
    ):
        self.particle_data, self.jet_data = self.getData(
            jet_type,
            data_dir,
            particle_features,
            jet_features,
            num_particles,
            split,
            split_fraction,
            seed,
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
        jet_type: Union[str, Set[str]] = "all",
        data_dir: str = "./",
        particle_features: List[str] = all_particle_features,
        jet_features: List[str] = all_jet_features,
        num_particles: int = 30,
        split: str = "all",
        split_fraction: List[float] = [0.7, 0.15, 0.15],
        seed: int = 42,
        download: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Downloads, if needed, and loads and returns JetNet data.

        Args:
            jet_type (Union[str, Set[str]], optional): individual type or set of types out of
                'g' (gluon), 't' (top quarks), 'q' (light quarks), 'w' (W bosons),
                or 'z' (Z bosons). "all" will get all types. Defaults to "all".
            data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
            particle_features (List[str], optional): list of particle features to retrieve. If empty
                or None, gets no particle features. Defaults to
                ``["etarel", "phirel", "ptrel", "mask"]``.
            jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
                gets no jet features. Defaults to
                ``["type", "pt", "eta", "mass", "num_particles"]``.
            num_particles (int, optional): number of particles to retain per jet, max of 150.
                Defaults to 30.
            split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
                to "train".
            split_fraction (List[float], optional): splitting fraction of training, validation,
                testing data respectively. Defaults to [0.7, 0.15, 0.15].
            seed (int, optional): PyTorch manual seed - important to use the same seed for all
                dataset splittings. Defaults to 42.
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in the ``data_dir`` directory. If dataset is already downloaded, it is not
                downloaded again. Defaults to False.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: particle data, jet data
        """
        assert num_particles <= cls.max_num_particles, (
            f"num_particles {num_particles} exceeds max number of "
            + f"particles in the dataset {cls.max_num_particles}"
        )
        jet_type = checkConvertElements(jet_type, cls.jet_types, ntype="jet type")
        particle_features, jet_features = checkStrToList(particle_features, jet_features)
        use_particle_features, use_jet_features = checkListNotEmpty(particle_features, jet_features)

        import h5py

        # Use JetNet150 if ``num_particles`` > 30
        use_150 = num_particles > 30

        particle_data = []
        jet_data = []

        for j in jet_type:
            dname = f"{j}{'150' if use_150 else ''}"

            hdf5_file = checkDownloadZenodoDataset(
                data_dir,
                dataset_name=dname,
                record_id=cls._zenodo_record_ids["150" if use_150 else "30"],
                key=f"{dname}.hdf5",
                download=download,
            )

            with h5py.File(hdf5_file, "r") as f:
                pf = (
                    np.array(f["particle_features"])[:, :num_particles]
                    if use_particle_features
                    else None
                )
                jf = np.array(f["jet_features"]) if use_jet_features else None

            if use_particle_features:
                # reorder if needed
                pf = getOrderedFeatures(pf, particle_features, cls.all_particle_features)

            if use_jet_features:
                # add class index as first jet feature
                class_index = cls.jet_types.index(j)
                jf = np.concatenate(
                    (
                        np.full([len(jf), 1], class_index),
                        jf[:, :3],
                        # max particles should be num particles
                        np.minimum(jf[:, 3:], num_particles),
                    ),
                    axis=1,
                )
                # reorder if needed
                jf = getOrderedFeatures(jf, jet_features, cls.all_jet_features)

            particle_data.append(pf)
            jet_data.append(jf)

        particle_data = np.concatenate(particle_data, axis=0) if use_particle_features else None
        jet_data = np.concatenate(jet_data, axis=0) if use_jet_features else None

        length = len(firstNotNoneElement(particle_data, jet_data))

        # shuffling and splitting into training and test
        lcut, rcut = getSplitting(length, split, cls.splits, split_fraction)

        np.random.seed(seed)
        randperm = np.random.permutation(length)

        if use_particle_features:
            particle_data = particle_data[randperm][lcut:rcut]

        if use_jet_features:
            jet_data = jet_data[randperm][lcut:rcut]

        return particle_data, jet_data

    def extra_repr(self) -> str:
        ret = f"Including {self.jet_type} jets"

        if self.split == "all":
            ret += "\nUsing all data (no split)"
        else:
            ret += (
                f"\nSplit into {self.split} data out of {self.splits} possible splits, "
                f"with splitting fractions {self.split_fraction}"
            )

        return ret
