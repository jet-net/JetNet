from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np

from .dataset import JetDataset
from .normalisations import NormaliseABC
from .utils import (
    checkConvertElements,
    checkDownloadZenodoDataset,
    checkListNotEmpty,
    checkStrToList,
    getOrderedFeatures,
)


class TopTagging(JetDataset):
    """
    PyTorch ``torch.unit.data.Dataset`` class for the Top Quark Tagging Reference dataset.

    If hdf5 files are not found in the ``data_dir`` directory then dataset will be downloaded
    from Zenodo (https://zenodo.org/record/2603256).

    Args:
        jet_type (Union[str, Set[str]], optional): individual type or set of types out of 'qcd' and
            'top'. Defaults to "all".
        data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
        particle_features (List[str], optional): list of particle features to retrieve. If empty
            or None, gets no particle features. Defaults to ``["E", "px", "py", "pz"]``.
        jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
            gets no jet features. Defaults to ``["type", "E", "px", "py", "pz"]``.
        particle_normalisation (NormaliseABC, optional): optional normalisation to apply to
            particle data. Defaults to None.
        jet_normalisation (NormaliseABC, optional): optional normalisation to apply to jet data.
            Defaults to None.
        particle_transform (callable, optional): A function/transform that takes in the particle
            data tensor and transforms it. Defaults to None.
        jet_transform (callable, optional): A function/transform that takes in the jet
            data tensor and transforms it. Defaults to None.
        num_particles (int, optional): number of particles to retain per jet, max of 200. Defaults
            to 200.
        split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
            to "train".
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in the ``data_dir`` directory. If dataset is already downloaded, it is not
            downloaded again. Defaults to False.
    """

    _zenodo_record_id = 2603256
    max_num_particles = 200

    jet_types = ["qcd", "top"]
    all_particle_features = ["E", "px", "py", "pz"]
    all_jet_features = ["type", "E", "px", "py", "pz"]
    splits = ["train", "valid", "test"]
    _split_key_mapping = {"train": "train", "valid": "val", "test": "test"}  # map to file name

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
        num_particles: int = max_num_particles,
        split: str = "train",
        download: bool = False,
    ):
        self.particle_data, self.jet_data = self.getData(
            jet_type, data_dir, particle_features, jet_features, num_particles, split, download
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

    @classmethod
    def getData(
        cls,
        jet_type: Union[str, Set[str]] = "all",
        data_dir: str = "./",
        particle_features: List[str] = all_particle_features,
        jet_features: List[str] = all_jet_features,
        num_particles: int = max_num_particles,
        split: str = "all",
        download: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Downloads, if needed, and loads and returns Top Quark Tagging data.

        Args:
            jet_type (Union[str, Set[str]], optional): individual type or set of types out of 'qcd'
                and 'top'. Defaults to "all".
            data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
            particle_features (List[str], optional): list of particle features to retrieve. If empty
                or None, gets no particle features. Defaults to ``["E", "px", "py", "pz"]``.
            jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
                gets no jet features. Defaults to ``["type", "E", "px", "py", "pz"]``.
            num_particles (int, optional): number of particles to retain per jet, max of 200.
                Defaults to 200.
            split (str, optional): dataset split, out of {"train", "valid", "test", "all"}. Defaults
                to "all".
            download (bool, optional): If True, downloads the dataset from the internet and
                puts it in the ``data_dir`` directory. If dataset is already downloaded, it is not
                downloaded again. Defaults to False.

        Returns:
            (Tuple[Optional[np.ndarray], Optional[np.ndarray]]): particle data, jet data
        """
        assert num_particles <= cls.max_num_particles, (
            f"num_particles {num_particles} exceeds max number of "
            + f"particles in the dataset {cls.max_num_particles}"
        )

        jet_type = checkConvertElements(jet_type, cls.jet_types, ntype="jet type")
        type_indices = [cls.jet_types.index(t) for t in jet_type]

        particle_features, jet_features = checkStrToList(particle_features, jet_features)
        use_particle_features, use_jet_features = checkListNotEmpty(particle_features, jet_features)
        split = checkConvertElements(split, cls.splits, ntype="splitting")

        import pandas as pd

        particle_data = []
        jet_data = []

        for s in split:
            hdf5_file = checkDownloadZenodoDataset(
                data_dir,
                dataset_name=cls._split_key_mapping[s],
                record_id=cls._zenodo_record_id,
                key=f"{cls._split_key_mapping[s]}.h5",
                download=download,
            )

            data = np.array(pd.read_hdf(hdf5_file, key="table"))

            # select only specified types of jets (qcd or top or both)
            jet_selector = np.sum([data[:, -1] == i for i in type_indices], axis=0).astype(bool)
            data = data[jet_selector]

            # exctract particle and jet features in the order specified by the class
            # ``feature_order`` variables
            total_particle_features = cls.max_num_particles * len(cls.all_particle_features)

            if use_particle_features:
                pf = data[:, :total_particle_features].reshape(
                    -1, cls.max_num_particles, len(cls.all_particle_features)
                )[:, :num_particles]

                # reorder if needed
                pf = getOrderedFeatures(pf, particle_features, cls.all_particle_features)
                particle_data.append(pf)

            if use_jet_features:
                jf = np.concatenate(
                    (data[:, -1:], data[:, total_particle_features : total_particle_features + 4]),
                    axis=-1,
                )

                # reorder if needed
                jf = getOrderedFeatures(jf, jet_features, cls.all_jet_features)
                jet_data.append(jf)

        particle_data = np.concatenate(particle_data, axis=0) if use_particle_features else None
        jet_data = np.concatenate(jet_data, axis=0) if use_jet_features else None

        return particle_data, jet_data

    def extra_repr(self) -> str:
        ret = f"Including {self.jet_type} jets"

        if self.split == "all":
            ret += "\nUsing all data (no split)"
        else:
            ret += f"\nSplit into {self.split} data out of {self.splits} possible splits"

        return ret
