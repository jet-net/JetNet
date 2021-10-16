from typing import List, Union, Optional
import torch
from torch import Tensor
import logging
from os.path import exists
import numpy as np


# TODO: allow for loading all three jet types together


class JetNet(torch.utils.data.Dataset):
    """
    PyTorch ``torch.utils.data.Dataset`` class for the JetNet dataset, shape is ``[num_jets, num_particles, num_features]``.

    Features, in order: ``[eta, phi, pt, mask]``.

    Dataset is downloaded from https://zenodo.org/record/5502543 if pt or csv file is not found in the ``data_dir`` directory.

    Args:
        jet_type (str): 'g' (gluon), 't' (top quarks), or 'q' (light quarks).
        data_dir (str): directory which contains (or in which to download) dataset. Defaults to "./" i.e. the working directory.
        download (bool): download the dataset, even if the csv file exists already. Defaults to False.
        num_particles (int): number of particles to use, has to be less than the total in JetNet (30). 0 means use all. Defaults to 0.
        normalize (bool): normalize features for training or not, using parameters defined below. Defaults to True.
        feature_norms (Union[float, List[float]]): max value to scale each feature to.
          Can either be a single float for all features, or a list of length ``num_features``. Defaults to 1.0.
        feature_shifts (Union[float, List[float]]): after scaling, value to shift feature by.
          Can either be a single float for all features, or a list of length ``num_features``. Defaults to 0.0.
        use_mask (bool): Defaults to True.
        train (bool): whether for training or testing. Defaults to True.
        train_fraction (float): fraction of data to use as training - rest is for testing. Defaults to 0.7.
        num_pad_particles (int): how many out of ``num_particles`` should be zero-padded. Defaults to 0.
        use_num_particles_jet_feature (bool): Store the # of particles in each jet as a jet-level feature.
          Only works if using mask. Defaults to True.
        noise_padding (bool): instead of 0s, pad extra particles with Gaussian noise. Only works if using mask. Defaults to False.
    """

    _fpnd_feature_maxes = [1.6211985349655151, 0.520724892616272, 0.8934717178344727, 1.0]
    _fpnd_feature_norms = 1.0
    _fpnd_feature_shifts = [0.0, 0.0, -0.5, 0.0]

    def __init__(
        self,
        jet_type: str,
        data_dir: str = "./",
        download: bool = False,
        num_particles: int = 0,
        normalize: bool = True,
        feature_norms: List[float] = [1.0, 1.0, 1.0, 1.0],
        feature_shifts: List[float] = [0.0, 0.0, -0.5, -0.5],
        use_mask: bool = True,
        train: bool = True,
        train_fraction: float = 0.7,
        num_pad_particles: int = 0,
        use_num_particles_jet_feature: bool = True,
        noise_padding: bool = False,
    ):
        assert jet_type in ["g", "t", "q"], "Invalid jet type"

        self.feature_norms = feature_norms
        self.feature_shifts = feature_shifts
        self.use_mask = use_mask
        # in the future there'll be more jet features such as jet pT and eta
        self.use_jet_features = (use_num_particles_jet_feature) and self.use_mask
        self.noise_padding = noise_padding and self.use_masks
        self.normalize = normalize

        pt_file = f"{data_dir}/{jet_type}_jets.pt"

        if not exists(pt_file) or download:
            self.download_and_convert_to_pt(data_dir, jet_type)

        logging.info("Loading dataset")
        dataset = self.load_dataset(pt_file, num_particles, num_pad_particles, use_mask)
        self.num_particles = num_particles if num_particles > 0 else dataset.shape[1]

        if self.use_jet_features:
            jet_features = self.get_jet_features(dataset, use_num_particles_jet_feature)

        logging.info(f"Loaded dataset {dataset.shape = }")
        if normalize:
            logging.info("Normalizing features")
            self.feature_maxes = self.normalize_features(dataset, feature_norms, feature_shifts)

        if self.noise_padding:
            dataset = self.add_noise_padding(dataset)

        tcut = int(len(dataset) * train_fraction)

        self.data = dataset[:tcut] if train else dataset[tcut:]
        if self.use_jet_features:
            self.jet_features = jet_features[:tcut] if train else jet_features[tcut:]

        logging.info("Dataset processed")

    def download_and_convert_to_pt(self, data_dir: str, jet_type: str):
        """
        Download jet dataset and convert and save to pytorch tensor

        Args:
            data_dir (str): directory in which to save file.
            jet_type (str): jet type to download, out of ``['g', 't', 'q']``.

        """
        import os

        os.system(f"mkdir -p {data_dir}")
        csv_file = f"{data_dir}/{jet_type}_jets.csv"

        if not exists(csv_file):
            logging.info(f"Downloading {jet_type} jets csv")
            self.download(jet_type, csv_file)

        logging.info(f"Converting {jet_type} jets csv to pt")
        self.csv_to_pt(data_dir, jet_type, csv_file)

    def download(self, jet_type: str, csv_file: str):
        """
        Downloads the ``jet_type`` jet csv from Zenodo and saves it as ``csv_file``.

        Args:
            jet_type (str): jet type to download, out of ``['g', 't', 'q']``.
            csv_file (str): path to save csv file.

        """
        import requests
        import sys

        records_url = "https://zenodo.org/api/records/5502543"
        r = requests.get(records_url).json()
        key = f"{jet_type}_jets.csv"
        file_url = next(item for item in r["files"] if item["key"] == key)["links"][
            "self"
        ]  # finding the url for the particular jet type dataset
        logging.info(f"{file_url = }")

        # modified from https://sumit-ghosh.com/articles/python-download-progress-bar/
        with open(csv_file, "wb") as f:
            response = requests.get(file_url, stream=True)
            total = response.headers.get("content-length")

            if total is None:
                f.write(response.content)
            else:
                downloaded = 0
                total = int(total)

                print("Downloading dataset")
                for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    done = int(50 * downloaded / total)
                    sys.stdout.write(
                        "\r[{}{}] {:.0f}%".format(
                            "█" * done, "." * (50 - done), float(downloaded / total) * 100
                        )
                    )
                    sys.stdout.flush()

        sys.stdout.write("\n")

    def csv_to_pt(self, data_dir: str, jet_type: str, csv_file: str):
        """
        Converts and saves downloaded csv file to pytorch tensor.

        Args:
            data_dir (str): directory in which to save file.
            jet_type (str): jet type to download, out of ``['g', 't', 'q']``.
            csv_file (str): path to csv file.

        """
        pt_file = f"{data_dir}/{jet_type}_jets.pt"
        torch.save(Tensor(np.loadtxt(csv_file).reshape(-1, 30, 4)), pt_file)

    def load_dataset(
        self, pt_file: str, num_particles: int, num_pad_particles: int = 0, use_mask: bool = True
    ) -> Tensor:
        """
        Load the dataset, optionally padding the particles.

        Args:
            pt_file (str): path to dataset .pt file.
            num_particles (int): number of particles per jet to load (has to be less than the number per jet in the dataset).
            num_pad_particles (int): out of ``num_particles`` how many are to be zero-padded. Defaults to 0.
            use_mask (bool): keep or remove the mask feature. Defaults to True.

        Returns:
            Tensor: dataset tensor of shape ``[num_jets, num_particles, num_features]``.

        """
        dataset = torch.load(pt_file).float()

        # only retain up to ``num_particles``, subtracting ``num_pad_particles`` since they will be padded below
        if 0 < num_particles - num_pad_particles < dataset.shape[1]:
            dataset = dataset[:, : num_particles - num_pad_particles, :]

        # pad with ``num_pad_particles`` particles
        if num_pad_particles > 0:
            dataset = torch.nn.functional.pad(dataset, (0, 0, 0, num_pad_particles), "constant", 0)

        if not use_mask:
            dataset = dataset[:, :, :-1]  # remove mask feature from dataset if not needed

        return dataset

    def get_jet_features(self, dataset: Tensor, use_num_particles_jet_feature: bool) -> Tensor:
        """
        Returns jet-level features. `Will be expanded to include jet pT and eta.`

        Args:
            dataset (Tensor):  dataset tensor of shape [N, num_particles, num_features], where the last feature is the mask.
            use_num_particles_jet_feature (bool): `Currently does nothing, in the future such bools will specify which jet features to use`.

        Returns:
            Tensor: jet features tensor of shape [N, num_jet_features].

        """
        jet_num_particles = (torch.sum(dataset[:, :, -1], dim=1) / self.num_particles).unsqueeze(1)
        logging.debug("{num_particles = }")
        return jet_num_particles

    @classmethod
    def normalize_features(
        self,
        dataset: Tensor,
        feature_norms: Union[float, List[float]] = 1.0,
        feature_shifts: Union[float, List[float]] = 0.0,
        fpnd: bool = False,
    ) -> Optional[List]:
        """
        Normalizes dataset features (in place), by scaling to ``feature_norms`` maximum and shifting by ``feature_shifts``.

        If the value in the List for a feature is None, it won't be scaled or shifted.

        If ``fpnd`` is True, will normalize instead to the same scale as was used for the ParticleNet training in https://arxiv.org/abs/2106.11535.

        Args:
            dataset (Tensor): dataset tensor of shape [N, num_particles, num_features].
            feature_norms (Union[float, List[float]]): max value to scale each feature to.
              Can either be a single float for all features, or a list of length ``num_features``. Defaults to 1.0.
            feature_shifts (Union[float, List[float]]): after scaling, value to shift feature by.
              Can either be a single float for all features, or a list of length ``num_features``. Defaults to 0.0.
            fpnd (bool): Normalize features for ParticleNet inference for the Frechet ParticleNet Distance metric.
              Will override `feature_norms`` and ``feature_shifts`` inputs. Defaults to False.

        Returns:
            Optional[List]: if ``fpnd`` is False, returns list of length ``num_features`` of max absolute values for each feature.
              Used for unnormalizing features.

        """
        num_features = dataset.shape[2]

        if not fpnd:
            feature_maxes = [
                float(torch.max(torch.abs(dataset[:, :, i]))) for i in range(num_features)
            ]
        else:
            feature_maxes = JetNet._fpnd_feature_maxes
            feature_norms = JetNet._fpnd_feature_norms
            feature_shifts = JetNet._fpnd_feature_shifts

        if isinstance(feature_norms, float):
            feature_norms = np.full(num_features, feature_norms)

        if isinstance(feature_shifts, float):
            feature_shifts = np.full(num_features, feature_shifts)

        logging.debug(f"{feature_maxes = }")

        for i in range(num_features):
            if feature_norms[i] is not None:
                dataset[:, :, i] /= feature_maxes[i]
                dataset[:, :, i] *= feature_norms[i]

            if feature_shifts[i] is not None and feature_shifts[i] != 0:
                dataset[:, :, i] += feature_shifts[i]

        if not fpnd:
            return feature_maxes

    def unnormalize_features(
        self,
        dataset: Union[Tensor, np.ndarray],
        ret_mask_separate: bool = True,
        is_real_data: bool = False,
        zero_mask_particles: bool = True,
        zero_neg_pt: bool = True,
    ):
        """
        Inverts the ``normalize_features()`` function on the input ``dataset`` array or tensor,
        plus optionally zero's the masked particles and negative pTs.
        Only applicable if dataset was normalized first i.e. ``normalize`` arg into JetNet instance is True.

        Args:
            dataset (Union[Tensor, np.ndarray]): Dataset to unnormalize.
            ret_mask_separate (bool): Return the jet and mask separately. Defaults to True.
            is_real_data (bool): Real or generated data. Defaults to False.
            zero_mask_particles (bool): Set features of zero-masked particles to 0. Not needed for real data. Defaults to True.
            zero_neg_pt (bool): Set pT to 0 for particles with negative pt. Not needed for real data. Defaults to True.

        Returns:
            Unnormalized dataset of same type as input. Either a tensor/array of shape ``[num_jets, num_particles, num_features (including mask)]``
            if ``ret_mask_separate`` is False, else a tuple with a tensor/array of shape ``[num_jets, num_particles, num_features (excluding mask)]``
            and another binary mask tensor/array of shape ``[num_jets, num_particles, 1]``.
        """
        if not self.normalize:
            raise RuntimeError("Can't unnormalize features if dataset has not been normalized.")

        num_features = dataset.shape[2]

        for i in range(num_features):
            if self.feature_shifts[i] is not None and self.feature_shifts[i] != 0:
                dataset[:, :, i] -= self.feature_shifts[i]

            if self.feature_norms[i] is not None:
                dataset[:, :, i] /= self.feature_norms[i]
                dataset[:, :, i] *= self.feature_maxes[i]

        mask = dataset[:, :, -1] >= 0.5 if self.use_mask else None

        if not is_real_data and zero_mask_particles and self.use_mask:
            dataset[~mask] = 0

        if not is_real_data and zero_neg_pt:
            dataset[:, :, 2][dataset[:, :, 2] < 0] = 0

        return dataset[:, :, :-1], mask if ret_mask_separate else dataset

    def add_noise_padding(self, dataset: Tensor):
        """Add Gaussian noise to zero-masked particles"""
        logging.debug(f"Pre-noise padded dataset: \n {dataset[:2, -10:]}")

        # up to 5 sigmas will be within ±1
        noise_padding = torch.randn((len(dataset), self.num_particles, dataset.shape[2] - 1)) / 5
        noise_padding[noise_padding > 1] = 1
        noise_padding[noise_padding < -1] = -1
        noise_padding[:, :, 2] /= 2.0  # pt is scaled between ±0.5

        mask = (dataset[:, :, 3] + 0.5).bool()
        noise_padding[mask] = 0  # only adding noise to zero-masked particles
        dataset += torch.cat(
            (noise_padding, torch.zeros((len(dataset), self.num_particles, 1))), dim=2
        )

        logging.debug("Post-noise padded dataset: \n {dataset[:2, -10:]}")

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.jet_features[idx] if self.use_jet_features else self.data[idx]
