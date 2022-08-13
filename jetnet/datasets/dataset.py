"""
Base classes for JetNet datasets.
"""

import torch
from torch import Tensor

from typing import List, Optional, Tuple, Any

from .normalisations import NormaliseABC
from .utils import checkStrToList, checkListNotEmpty, firstNotNoneElement


class JetDataset(torch.utils.data.Dataset):
    """
    Base class for jet datasets.
    Inspired by https://pytorch.org/vision/main/generated/torchvision.datasets.VisionDataset.html

    Args:
        data_dir (str): directory where dataset is or will be stored.
        particle_normalisation (Optional[NormaliseABC], optional): optional normalisation for
            particle-level features. Defaults to None.
        jet_normalisation (Optional[NormaliseABC], optional): optional normalisation for jet-level
            features. Defaults to None.
    """

    _repr_indent = 4

    particle_data = None
    jet_data = None

    def __init__(
        self,
        data_dir: str = "./",
        particle_features: Optional[List[str]] = None,
        jet_features: Optional[List[str]] = None,
        particle_normalisation: Optional[NormaliseABC] = None,
        jet_normalisation: Optional[NormaliseABC] = None,
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
                self.particle_normalisation.derive_dataset_features(self.particle_data)
                self.particle_data = self.particle_normalisation(self.particle_data)

            self.particle_data = Tensor(self.particle_data)

        if self.use_jet_features:
            if self.jet_normalisation is not None:
                self.jet_normalisation.derive_dataset_features(self.jet_data)
                self.jet_data = self.jet_normalisation(self.jet_data)

            self.jet_data = Tensor(self.jet_data)

    @staticmethod
    def getData(**opts) -> Any:
        """Static method to download and return numpy arrays of the data"""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index
        Returns:
            (Any): particle and/or jet data
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(firstNotNoneElement(self.particle_data, self.jet_data))

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]

        if self.data_dir is not None:
            body.append(f"Data location: {self.data_dir}")

        body += self.extra_repr().splitlines()

        if self.particle_features is not None:
            body += [f"Particle features: {self.particle_features}"]

        if self.jet_features is not None:
            body += [f"Jet features: {self.jet_features}"]

        if self.particle_normalisation is not None:
            body += [f"Particle normalisation: {self.particle_normalisation}"]

        if self.jet_normalisation is not None:
            body += [f"Jet normalisation: {self.jet_normalisation}"]

        lines = [head] + [" " * self._repr_indent + line for line in body]

        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""
