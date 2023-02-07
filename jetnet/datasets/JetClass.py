from typing import Callable, List, Set, Union, Optional, Tuple
import numpy as np
import logging
import uproot
import os
from utils_jetclass import *

class JetClass:
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

    jet_type = ["HtoBB", "HtoCC", "HtoGG", "HtoWW", "HtoWW2Q1L", "HtoWW4Q", "TTBar", "TTBarLep", 
                "WtoQQ", "ZJetstoNuNu", "ZtoQQ"]
    all_particle_features = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi", "part_d0val", "part_d0err", "part_dzval",
                             "part_dzerr", "part_charge", "part_isChargedHadron", "part_isNeutralHadron", "part_isPhoton", "part_isElectron", "part_isMuon"]
    all_jet_features = ["jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_nparticles", "jet_sdmass", "jet_tau1", "jet_tau2", "jet_tau3", "jet_tau4"]
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
            jet_type,
            data_dir,
            particle_features,
            jet_features
        )

        super().__init__(
            data_dir=data_dir,
            particle_features=particle_features,
            jet_features=jet_features,
        )
        self.split = split
        self.split_fraction = split_fraction

    @classmethod
    def getData(self,jet_type, data_dir, particle_features, jet_features):
        dataset_name = "JetClass Validation Set"
        file_download_name = "Val_5M"
        key = "JetClass_Pythia_val_5M.tar"
        record_id = 6619768
        jet_matrix = np.zeros((1, 100000))
        particle_matrix = np.zeros((1, 136))
        file_path = checkDownloadZenodoDataset(data_dir, dataset_name, record_id, key, file_download_name)
        print("Processing Data: ...")
        for jet_file in os.listdir(file_path):
            f = os.path.join(file_path, jet_file)
            for jet in jet_type:
                if jet in f:
                    open_file = uproot.open(f)
                    branch = open_file['tree']
                    for i in branch.keys():
                        for feature in jet_features:
                            if feature in i:
                                arr = branch[i].array()
                                arr = np.array(arr)
                                jet_matrix = np.vstack([jet_matrix, arr])
                        for particle in particle_features:
                            if particle in i:
                                arr_awk = branch[i].array()
                                awk_list = list(arr_awk)
                                zero_pad_arr = zero_padding(awk_list)
                                length_curr = findMaxLengthList(zero_pad_arr)
                                length_matrix = findMaxLengthList(particle_matrix)
                                zeros = np.zeros(100001)
                                if (length_curr > length_matrix) :
                                    zeros = np.zeros(100001)
                                    diff = length_curr - length_matrix
                                    for i in range(diff):
                                        particle_matrix = np.column_stack((particle_matrix,zeros))
                                elif (length_curr < length_matrix):
                                    zeros = np.zeros(100000)
                                    diff = length_matrix - length_curr
                                    for i in range(diff):
                                        zero_pad_arr = np.column_stack((zero_pad_arr,zeros))
                                particle_matrix = np.vstack([particle_matrix, zero_pad_arr])
                                updated_particle_matrix = np.delete(particle_matrix, 0 , axis = 0)   
                
        updated_jet_matrix = np.delete(jet_matrix, 0 , axis = 0)                
        dim1 = updated_jet_matrix.shape[0]
        dim2 = updated_jet_matrix.shape[1]
        dim_res = dim1/len(jet_features)
        dim = int(dim_res * dim2)
        return updated_jet_matrix.reshape(dim,len(jet_features)) , updated_particle_matrix
