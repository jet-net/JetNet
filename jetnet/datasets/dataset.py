"""
Base classes for JetNet datasets.
"""

from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor

from .normalisations import NormaliseABC
from .utils import checkListNotEmpty, checkStrToList, firstNotNoneElement


class JetDataset(torch.utils.data.Dataset):
    """
    Base class for jet datasets.
    Inspired by https://pytorch.org/vision/main/generated/torchvision.datasets.VisionDataset.html

    Args:
        data_dir (str): directory where dataset is or will be stored.
        particle_features (List[str], optional): list of particle features to retrieve. If empty
            or None, gets no particle features. Should default to all.
        jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
            gets no particle features. Should default to all.
        particle_normalisation (Optional[NormaliseABC], optional): optional normalisation for
            particle-level features. Defaults to None.
        jet_normalisation (Optional[NormaliseABC], optional): optional normalisation for jet-level
            features. Defaults to None.
        particle_transform (callable, optional): A function/transform that takes in the particle
            data tensor and transforms it. Defaults to None.
        jet_transform (callable, optional): A function/transform that takes in the jet
            data tensor and transforms it. Defaults to None.
        num_particles (int, optional): max number of particles to retain per jet. Defaults to None.
    """

    _repr_indent = 4

    particle_data = None
    jet_data = None
    max_num_particles = None

    def __init__(
        self,
        data_dir: str = "./",
        particle_features: Optional[List[str]] = None,
        jet_features: Optional[List[str]] = None,
        particle_normalisation: Optional[NormaliseABC] = None,
        jet_normalisation: Optional[NormaliseABC] = None,
        particle_transform: Optional[Callable] = None,
        jet_transform: Optional[Callable] = None,
        num_particles: Optional[int] = None,
    ):
        self.data_dir = data_dir

        self.particle_features, self.jet_features = checkStrToList(particle_features, jet_features)
        self.use_particle_features, self.use_jet_features = checkListNotEmpty(
            particle_features, jet_features
        )

        self.particle_normalisation = particle_normalisation
        self.jet_normalisation = jet_normalisation

        if self.use_particle_features:
            if self.particle_normalisation is not None:
                if self.particle_normalisation.features_need_deriving():
                    self.particle_normalisation.derive_dataset_features(self.particle_data)
                self.particle_data = self.particle_normalisation(self.particle_data)

        if self.use_jet_features:
            if self.jet_normalisation is not None:
                if self.jet_normalisation.features_need_deriving():
                    self.jet_normalisation.derive_dataset_features(self.jet_data)
                self.jet_data = self.jet_normalisation(self.jet_data)

        self.particle_transform = particle_transform
        self.jet_transform = jet_transform

        self.num_particles = num_particles

    @classmethod
    def getData(**opts) -> Any:
        """Class method to download and return numpy arrays of the data"""
        raise NotImplementedError

    def __getitem__(self, index) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """
        Gets data and if needed transforms it.

        Args:
            index (int): Index

        Returns:
            (Tuple[Optional[Tensor], Optional[Tensor]]): particle, jet data
        """

        if self.use_particle_features:
            particle_data = self.particle_data[index]

            if self.particle_transform is not None:
                particle_data = self.particle_transform(particle_data)

            particle_data = Tensor(particle_data)
        else:
            particle_data = []

        if self.use_jet_features:
            jet_data = self.jet_data[index]

            if self.jet_transform is not None:
                jet_data = self.jet_transform(jet_data)

            jet_data = Tensor(jet_data)
        else:
            jet_data = []

        return particle_data, jet_data

    def __len__(self) -> int:
        return len(firstNotNoneElement(self.particle_data, self.jet_data))

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]

        if self.data_dir is not None:
            body.append(f"Data location: {self.data_dir}")

        body += self.extra_repr().splitlines()

        if self.particle_features is not None:
            bstr = f"Particle features: {self.particle_features}"
            if self.num_particles is not None:
                bstr += f", max {self.num_particles} particles per jet"

            body += [bstr]

        if self.jet_features is not None:
            body += [f"Jet features: {self.jet_features}"]

        if self.particle_normalisation is not None:
            body += [f"Particle normalisation: {self.particle_normalisation}"]

        if self.jet_normalisation is not None:
            body += [f"Jet normalisation: {self.jet_normalisation}"]

        if self.particle_transform is not None:
            body += [f"Particle transform: {self.particle_transform}"]

        if self.jet_transform is not None:
            body += [f"Jet transform: {self.jet_transform}"]

        lines = [head] + [" " * self._repr_indent + line for line in body]

        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""
