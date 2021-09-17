# energyflow needs to be imported before pytorch because of https://github.com/pkomiske/EnergyFlow/issues/24
from energyflow.emd import emds
from .dataset import JetNet
from typing import Union, Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import numpy as np

from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

import logging


def cov_mmd(
    real_jets: Union[Tensor, np.array],
    gen_jets: Union[Tensor, np.array],
    num_eval_samples: int = 100,
    num_batches: int = 100,
    in_energyflow_format: bool = False,
) -> Tuple[float, float]:
    """
    Calculate coverage and MMD between real and generated jets, using the Energy Mover's Distance as the distance metric.

    Args:
        real_jets (Union[Tensor, np.array]): tensor or array of real jets, in either JetNet format [eta, phi, pt, mask] or energyflow format [pt, eta, phi] (specified via `in_energyflow_format` optional bool).
        gen_jets (Union[Tensor, np.array]): tensor or array of generated jets, same format as real_jets.
        num_eval_samples (int): number of jets out of the real and gen jets each between which to evaluate COV and MMD. Defaults to 100.
        num_batches (int): number of different batches to calculate COV and MMD and average over. Defaults to 100.
        in_energyflow_format (bool): are the jets in energyflow or JetNet format. Defaults to False.

    Returns:
        float: coverage, averaged over `num_batches`.
        float: MMD, averaged over `num_batches`.

    """
    if isinstance(real_jets, Tensor):
        real_jets = real_jets.cpu().detach().numpy()

    if isinstance(gen_jets, Tensor):
        gen_jets = gen_jets.cpu().detach().numpy()

    if not in_energyflow_format:
        # convert from JetNet [eta, phi, pt, mask] format to energyflow [pt, eta, phi]
        real_jets = real_jets[:, :, [2, 0, 1]]
        gen_jets = gen_jets[:, :, [2, 0, 1]]

    covs = []
    mmds = []

    for j in range(num_batches):
        real_rand = rng.choice(len(real_jets), size=num_eval_samples)
        gen_rand = rng.choice(len(gen_jets), size=num_eval_samples)

        real_rand_sample = real_jets[real_rand]
        gen_rand_sample = gen_jets[gen_rand]

        dists = emds(gen_rand_sample, real_rand_sample)  # 2D array of emds, with shape (len(gen_rand_sample), len(real_rand_sample))

        mmds.append(np.mean(np.min(dists, axis=0)))  # for MMD, for each gen jet find the closest real jet and average EMDs
        covs.append(
            np.unique(np.argmin(dists, axis=1)).size / num_eval_samples
        )  # for coverage, for each real jet find the closest gen jet and get the number of unique matchings

    return np.mean(covs), np.mean(mmds)
