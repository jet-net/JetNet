from typing import List, Union, Optional
import torch
import torchvision

from torch import Tensor
import numpy as np

import logging
from os.path import exists

class TopTagging(torch.utils.data.Dataset) :


    def __init__(
        self,
        is_Quark: str,
        data_dir: str = "./",
        download: bool = False,
        num_particles: int = 30,
        normalize: bool = True,
        feature_norms: List[float] = [1.0, 1.0, 1.0, 1.0],
        feature_shifts: List[float] = [0.0, 0.0, -0.5, -0.5],
        use_mask: bool = True,
       # train: bool = True,
        train_fraction: float = 0.7,
        num_pad_particles: int = 0,
        use_num_particles_jet_feature: bool = True,
        noise_padding: bool = False,
    ):
        #pt_file = f"{data_dir}/{is_Quark}.pt"
        assert is_Quark in ["top", "qcd"], "Invalid jet type"
        dataset_type = input("Enter 'test' for testing data, 'train' for training data and 'val' for validation data: ")
        if dataset_type == 'test':
            self.download_and_convert_to_pt(data_dir, dataset_type)

        elif dataset_type == 'train':
            self.download_and_convert_to_pt(data_dir, dataset_type)

        elif dataset_type == 'val' :
            self.download_and_convert_to_pt(data_dir, dataset_type)
        
        pt_file = f"{data_dir}/{dataset_type}.pt"
        #pt_file = f"{dataset_type}.pt"


        # self.feature_norms = feature_norms
        # self.feature_shifts = feature_shifts
        # self.use_mask = use_mask
	    # # in the future there will be more jet features such as jet pT and eta
        # self.use_jet_features = use_num_particles_jet_feature and self.use_mask
        # self.noise_padding = noise_padding and self.use_maskse
        # self.normalize = normalizw


        if not exists(pt_file) or download:
             self.download_and_convert_to_pt(data_dir, dataset_type)
        
        logging.info("Loading Dataset")
        dataset = self.load_dataset(pt_file, num_particles)
        self.num_particles = num_particles if num_particles > 0 else dataset.shape[1]

        #if self.use_jet_features:
           # jet_features = self.get_jet_features(dataset,use_num_particles_jet_feature)
        
        logging.info(f"Loaded dataset {dataset.shape = }")
        #if normalize:
            #logging.info("Normalizing features")
            #self.feature_maxes = self.normalize_features(dataset,feature_norms, feature_shifts)

        #if self.noise_padding:
            #dataset = self.add_noise_padding(dataset)
        
        self.data = dataset_type

    def download_and_convert_to_pt(self, data_dir: str, dataset_type: str):
            """
            Download jet dataset and convert and save to pytorch tensor.
            Args:
                data_dir (str): directory in which to save file.
                is_Quark (str): jet type to download, out of ``['g', 't', 'q']``.
                use_150 (bool): download JetNet150 or JetNet. Defaults to False.
            """
            import os

            os.system(f"mkdir -p {data_dir}")
            h5_file = f"{data_dir}/{dataset_type}.h5"

            if not exists(h5_file):
                logging.info(f"Downloading {dataset_type} quarks h5 file")
                self.download(dataset_type, h5_file)

            logging.info(f"Converting {dataset_type} jets h5 to pt")
            self.h5_to_pt(data_dir, dataset_type, h5_file)

    #download function
    def download(self, dataset_type: str, h5_file: str):
        """
        Downloads the ``jet_type`` jet hdf5 from Zenodo and saves it as ``hdf5_file``.

        Args:
            is_Quark (str): jet type to download, out of ``['top', 'qcd']``.
            h5_file (str): path to save hdf5 file.

        """
        import requests
        import sys

        record_id = 2603256
        records_url = f"https://zenodo.org/api/records/{record_id}"
        r = requests.get(records_url).json()
        key = f"{dataset_type}.h5"

        # finding the url for the particular jet type dataset
        #print(r)
        #print(key)

        file_url = next(item for item in r["files"] if item["key"] == key)["links"]["self"]
        logging.info(f"{file_url = }")

        # modified from https://sumit-ghosh.com/articles/python-download-progress-bar/
        with open(h5_file, "wb") as f:
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
        
    def h5_to_pt(self, data_dir: str, dataset_type: str, h5_file: str):
        """
        Converts and saves downloaded hdf5 file to pytorch tensor.

        Args:
            data_dir (str): directory in which to save file.
            jet_type (str): jet type to download, out of ``['g', 't', 'q']``.
            hdf5_file (str): path to hdf5 file.

        """
        #import h5py

        pt_file = f"{data_dir}/{dataset_type}.pt"

        #with h5py.File(h5_file, "r") as f:
            #torch.save(Tensor(np.float64(f["table"]["_i_table"]["index"]["abounds"])), pt_file)

        import pandas as pd
        import torch

        pdtable = pd.read_hdf(h5_file, key = "table")

        torch.save(Tensor(pdtable.values),pt_file)

        #torch.tensor(pdtable.values)

        

    def load_dataset(
        self, pt_file: str, num_particles: int
    ) -> Tensor:
        
        """
        Load the dataset, optionally padding the particles.

        Args:
            pt_file (str): path to dataset .pt file.
            num_particles (int): number of particles per jet to load
              (has to be less than the number per jet in the dataset).
            num_pad_particles (int): out of ``num_particles`` how many are to be zero-padded.
              Defaults to 0.
            use_mask (bool): keep or remove the mask feature. Defaults to True.

        Returns:
            Tensor: dataset tensor of shape ``[num_jets, num_particles, num_features]``.

        """
        dataset = torch.load(pt_file).float()
        print(dataset)


        # only retain up to ``num_particles``,
        # subtracting ``num_pad_particles`` since they will be padded below
        #print(num_particles)
        #if 0 < num_particles < dataset.shape[1]:
        #dataset = dataset[:, : num_particles, :]
        dataset = dataset[:num_particles]

        # pad with ``num_pad_particles`` particles
        #if num_pad_particles > 0:
           # dataset = torch.nn.functional.pad(dataset, (0, 0, 0, num_pad_particles), "constant", 0)

        #if not use_mask:
            # remove mask feature from dataset if not needed
            #dataset = dataset[:, :, : self._num_non_mask_features]

        return dataset[0:20]
    def get_jet_features(self, dataset: Tensor, use_num_particles_jet_feature: bool) -> Tensor:
        """
        Returns jet-level features. `Will be expanded to include jet pT and eta.`

        Args:
            dataset (Tensor):  dataset tensor of shape [N, num_particles, num_features],
              where the last feature is the mask.
            use_num_particles_jet_feature (bool): `Currently does nothing,
              in the future such bools will specify which jet features to use`.

        Returns:
            Tensor: jet features tensor of shape [N, num_jet_features].

        """
        jet_num_particles = (torch.sum(dataset[:, :, -1], dim=1) / self.num_particles).unsqueeze(1)
        logging.debug("{num_particles = }")
        return jet_num_particles

    @classmethod

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.jet_features[idx] if self.use_jet_features else self.data[idx]


