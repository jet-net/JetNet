import logging
import os
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
import uproot
from utils import *

from .dataset import JetDataset


class JetClass(JetDataset):
    """
    PyTorch ``torch.unit.data.Dataset`` class for the JetClass dataset.
    If root files are not found in the ``data_dir`` directory then dataset will be downloaded
    from Zenodo (https://zenodo.org/record/6619768).
    Args:
        jet_type (Union[str, Set[str]], optional): individual type or set of types out of 'HToBB' ,
            "HtoCC", "HtoGG", "HtoWW", "HtoWW2Q1L", "HtoWW4Q", "TTBar", "TTBarLep", "WtoQQ",
             "ZJetstoNuNu", "ZtoQQ" ). "all" will get all types. Defaults to "all".
        data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
        particle_features (List[str], optional): list of particle features to retrieve. If empty
            or None, gets no particle features. Defaults to
            `` ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi", "part_d0val",
            "part_d0err", "part_dzval", "part_dzerr", "part_charge", "part_isChargedHadron",
            "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon"]``.
        jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
            gets no jet features. Defaults to
            ``["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles", "jet_sdmass", "jet_tau1",
               "jet_tau2", "jet_tau3", "jet_tau4"]``.
    """

    zenodo_record_id = 6619768

    jet_type = [
        "HtoBB",
        "HtoCC",
        "HtoGG",
        "HtoWW",
        "HtoWW2Q1L",
        "HtoWW4Q",
        "TTBar",
        "TTBarLep",
        "WtoQQ",
        "ZJetstoNuNu",
        "ZtoQQ",
    ]
    all_particle_features = [
        "part_px",
        "part_py",
        "part_pz",
        "part_energy",
        "part_deta",
        "part_dphi",
        "part_d0val",
        "part_d0err",
        "part_dzval",
        "part_dzerr",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    ]
    all_jet_features = [
        "jet_pt",
        "jet_eta",
        "jet_phi",
        "jet_energy",
        "jet_nparticles",
        "jet_sdmass",
        "jet_tau1",
        "jet_tau2",
        "jet_tau3",
        "jet_tau4",
    ]
    splits = ["train", "valid", "test", "all"]

    def __init__(
        self,
        jet_type: Union[str, Set[str]] = "all",
        data_dir: str = "./",
        particle_features: List[str] = all_particle_features,
        jet_features: List[str] = all_jet_features,
        split: str = "train",
        split_fraction: List[float] = [0.7, 0.15, 0.15],
        seed: int = 42,
    ):
        self.particle_data, self.jet_data = self.getData(
            jet_type, data_dir, particle_features, jet_features
        )

        super().__init__(
            data_dir=data_dir,
            particle_features=particle_features,
            jet_features=jet_features,
        )
        self.split = split
        self.split_fraction = split_fraction

    @classmethod
    def getData(self, jet_type, data_dir, particle_features, jet_features):
        """
        Downloads JetClass dataset from zenodo if dataset is not already downloaded in
        user specified data directory. Loads and returns the JetClass data in the form a
        multidimensional NumPy array.

        Args:
            jet_type (Union[str, Set[str]]): individual type or set of types out of 'HToBB' ,
                "HtoCC", "HtoGG", "HtoWW", "HtoWW2Q1L", "HtoWW4Q", "TTBar", "TTBarLep", "WtoQQ",
                "ZJetstoNuNu", "ZtoQQ" ).
                data_dir (str, optional):
                data_dir (str, optional): directory in which data is (to be) stored. Defaults to "./".
            particle_features (List[str], optional): list of particle features to retrieve. If empty
                or None, gets no particle features. Defaults to
                `` ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi", "part_d0val",
                "part_d0err", "part_dzval", "part_dzerr", "part_charge", "part_isChargedHadron",
                "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon"]``.
            jet_features (List[str], optional): list of jet features to retrieve.  If empty or None,
                gets no jet features. Defaults to ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles", "jet_sdmass", "jet_tau1",
                "jet_tau2", "jet_tau3", "jet_tau4"].
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: jet data, particle data

        """

        dataset_name = "JetClass Validation Set"
        file_download_name = "Val_5M"
        key = "JetClass_Pythia_val_5M.tar"
        record_id = 6619768
        # Initializing empty matrix to return jet data
        jet_matrix = np.zeros((1, 100000))
        # Initializing empty matrix to return particle data
        particle_matrix = np.zeros((1, 136))
        # Extracting the file path
        file_path = checkDownloadZenodoDataset(
            data_dir, dataset_name, record_id, key, file_download_name
        )
        print("Processing Data: ...")
        # Looping thrpugh each root file in directory
        for jet_file in os.listdir(file_path):
            f = os.path.join(file_path, jet_file)
            for jet in jet_type:
                # Checking if user specified jet type(s) is in one of the filepaths of our directory
                if jet in f:
                    # opening root file that contains user specified jet type
                    open_file = uproot.open(f)
                    # root file contains one branch 'tree'
                    branch = open_file["tree"]
                    # looping through keys in the tree branch
                    for i in branch.keys():
                        for feature in jet_features:
                            # checking if user specified jet feature type(s) are part of the keys
                            if feature in i:
                                arr = branch[i].array()
                                # Converting the array to a numpy array
                                arr = np.array(arr)
                                # Concatenating np array to jet matrix
                                jet_matrix = np.vstack([jet_matrix, arr])
                        for particle in particle_features:
                            # checking if user specified particle feature type(s) are part of the keys
                            if particle in i:
                                arr_awk = branch[i].array()
                                # Converting awkward level array to a list
                                awk_list = list(arr_awk)
                                # takes in the 'awk_list' and zero pads the sublists in order to match dimensions
                                zero_pad_arr = zero_padding(awk_list)
                                # finds the max length sub list
                                length_curr = findMaxLengthList(zero_pad_arr)
                                length_matrix = findMaxLengthList(particle_matrix)
                                zeros = np.zeros(100001)
                                if length_curr > length_matrix:
                                    zeros = np.zeros(100001)
                                    diff = length_curr - length_matrix
                                    for i in range(diff):
                                        particle_matrix = np.column_stack((particle_matrix, zeros))
                                elif length_curr < length_matrix:
                                    zeros = np.zeros(100000)
                                    diff = length_matrix - length_curr
                                    for i in range(diff):
                                        zero_pad_arr = np.column_stack((zero_pad_arr, zeros))
                                particle_matrix = np.vstack([particle_matrix, zero_pad_arr])
                                # removing extra row from 'particle_matrix'
                                updated_particle_matrix = np.delete(particle_matrix, 0, axis=0)
        # removing extra row from 'jet_matrix
        updated_jet_matrix = np.delete(jet_matrix, 0, axis=0)
        # reshaping Jet Matrix
        dim1 = updated_jet_matrix.shape[0]
        dim2 = updated_jet_matrix.shape[1]
        dim_res = dim1 / len(jet_features)
        dim = int(dim_res * dim2)
        return updated_jet_matrix.reshape(dim, len(jet_features)), updated_particle_matrix
