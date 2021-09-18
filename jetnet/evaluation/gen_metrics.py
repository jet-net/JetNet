# energyflow needs to be imported before pytorch because of https://github.com/pkomiske/EnergyFlow/issues/24
from energyflow.emd import emds
from energyflow import EFPSet

from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor

from jetnet import utils

from scipy.stats import wasserstein_distance

# from .particlenet import ParticleNet

from tqdm import tqdm

from os import path
import sys


rng = np.random.default_rng()


# TODOs:
# - Functionality for adding FPND for new datasets
# - Cartesian coordinates

#
#
# def get_mu2_sigma2(args, C, X_loaded, fullpath):
#     """calculates means (mu) and covariance matrix (sigma) of activations of classifier C wrt real data"""
#
#     logging.info("Getting mu2, sigma2")
#
#     C.eval()
#     for i, jet in tqdm(enumerate(X_loaded), total=len(X_loaded)):
#         if i == 0:
#             activations = C(jet[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach()
#         else:
#             activations = torch.cat((C(jet[0][:, :, :3].to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)
#
#     activations = activations.numpy()
#
#     mu = np.mean(activations, axis=0)
#     sigma = np.cov(activations, rowvar=False)
#
#     np.savetxt(fullpath + "mu2.txt", mu)
#     np.savetxt(fullpath + "sigma2.txt", sigma)
#
#     return mu, sigma
#
#
# def load(args, X_loaded=None):
#     """loads pre-trained ParticleNet classifier and either calculates or loads from directory the means and covariance matrix of the activations wrt real data"""
#
#     C = ParticleNet(args.num_hits, args.node_feat_size, device=args.device).to(args.device)
#     C.load_state_dict(torch.load(args.evaluation_path + "C_state_dict.pt", map_location=args.device))
#
#     fullpath = args.evaluation_path + args.jets
#     logging.debug(fullpath)
#     if path.exists(fullpath + "mu2.txt"):
#         mu2 = np.loadtxt(fullpath + "mu2.txt")
#         sigma2 = np.loadtxt(fullpath + "sigma2.txt")
#     else:
#         mu2, sigma2 = get_mu2_sigma2(args, C, X_loaded, fullpath)
#
#     return (C, mu2, sigma2)
#
# def get_fpnd(args, C, gen_out, mu2, sigma2):
#     """calculates Frechet ParticleNet Distance of generated samples ``gen_out``"""
#
#     logging.info("Evaluating FPND")
#
#     gen_out_loaded = DataLoader(TensorDataset(torch.tensor(gen_out)), batch_size=args.fpnd_batch_size)
#
#     logging.info("Getting ParticleNet Acivations")
#     C.eval()
#     for i, gen_jets in tqdm(enumerate(gen_out_loaded), total=len(gen_out_loaded)):
#         gen_jets = gen_jets[0]
#         if args.mask:
#             mask = gen_jets[:, :, 3:4] >= 0
#             gen_jets = (gen_jets * mask)[:, :, :3]
#         if i == 0:
#             activations = C(gen_jets.to(args.device), ret_activations=True).cpu().detach()
#         else:
#             activations = torch.cat((C(gen_jets.to(args.device), ret_activations=True).cpu().detach(), activations), axis=0)
#
#     activations = activations.numpy()
#
#     mu1 = np.mean(activations, axis=0)
#     sigma1 = np.cov(activations, rowvar=False)
#
#     fpnd = utils.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
#     logging.info("PFND: " + str(fpnd))
#
#     return fpnd
#
#
# # this is a pointer to the module object instance itself. from https://stackoverflow.com/a/35904211/3759946
# eval_module = sys.modules[__name__]
# # for saving fpnd objects after the first loading
# eval_module.fpnd_dict = {
#     'NUM_SAMPLES': 50000
# }
#
#
# def fpnd(gen_jets: Union[Tensor, np.array], jet_type: str, dataset_name: str = 'JetNet'):
#     """
#     Plan for this function:
#     - first time called, loads the network and mu/sigma and saves in eval_module.fpnd_dict. after that keeps reusing from the dict
#     - can specify which dataset to use
#     - normalization will come from dataset's `normalize_features` function, which will include a special `fpnd` arg to normalize it the same way as was used for training.
#     - eventually build functionality for people to make fpnd for different datasets. this is complicated so LOWEST PRIORITY.
#     """
#     fpnd = None
#     return fpnd


def w1p(
    jets1: Union[Tensor, np.array],
    jets2: Union[Tensor, np.array],
    use_mask: bool = True,
    num_particle_features: int = 0,
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distances between particle features of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.array]): Tensor or array of jets, of shape ``[num_jets, num_particles_per_jet, num_features_per_particle]`` with an optional last binary mask feature per particle.
        jets2 (Union[Tensor, np.array]): Tensor or array of jets, of same format as ``jets1``.
        use_mask (bool): Use the last binary mask feature to ignore 0-masked particles. Defaults to True.
        num_particle_features (int): Will return W1 scores of the first ``num_particle_features`` particle features. If 0, will calcualte for all, excluding the optional mask feature if ``use_mask`` is True. Defaults to 0.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement. Defaults to 10000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        average_over_features (bool): Average over the particle features to return a single W1-P score. Defaults to True.
        return_std (bool): Return the standard deviation as well of the W1 scores over the ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[Union[float, np.array], Union[float, np.array]]:
        - **Union[float, np.array]**: if ``average_over_features`` is True, float of average W1 scores for each particle feature, first averaged over ``num_batches``,
          else array of length ``num_particle_features`` containing average W1 scores for each feature
        - **Union[float, np.array]** `(optional, only if ``return_std`` is True)`: if ``average_over_features`` is True, float of standard deviation of all W1 scores for each particle feature, first calculated over ``num_batches`` then propagated for the final average,
          else array of length ``num_particle_features`` containing standard deviation W1 scores for each feature

    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if num_particle_features <= 0:
        num_particle_features = jets1.shape[2] - int(use_mask)

    assert num_particle_features <= jets1.shape[2], "more particle features requested than were inputted"
    assert num_particle_features <= jets2.shape[2], "more particle features requested than were inputted"

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    w1s = []

    for j in range(num_batches):
        rand1 = rng.choice(len(jets1), size=num_eval_samples)
        rand2 = rng.choice(len(jets2), size=num_eval_samples)

        rand_sample1 = jets1[rand1]
        rand_sample2 = jets2[rand2]

        if use_mask:
            mask1 = rand_sample1[:, :, -1].astype(bool)
            mask2 = rand_sample2[:, :, -1].astype(bool)

            parts1 = rand_sample1[:, :, :num_particle_features][mask1]
            parts2 = rand_sample2[:, :, :num_particle_features][mask2]
        else:
            parts1 = rand_sample1[:, :, :num_particle_features].reshape(-1, num_particle_features)
            parts2 = rand_sample2[:, :, :num_particle_features].reshape(-1, num_particle_features)

        w1 = [wasserstein_distance(parts1[:, i], parts2[:, i]) for i in range(num_particle_features)]
        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    if average_over_features:
        return np.mean(means), np.linalg.norm(stds) if return_std else np.mean(means)
    else:
        return means, stds if return_std else means


def w1m(
    jets1: Union[Tensor, np.array],
    jets2: Union[Tensor, np.array],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distance between masses of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.array]): Tensor or array of jets, of shape ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``
        jets2 (Union[Tensor, np.array]): Tensor or array of jets, of same format as ``jets1``.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement. Defaults to 10000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        return_std (bool): Return the standard deviation as well of the W1 scores over the ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[float, float]:
        - **float**: W1 mass score, averaged over ``num_batches``.
        - **float** `(optional, only if ``return_std`` is True)`: standard deviation of W1 mass scores over ``num_batches``.

    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    masses1 = utils.jet_features(jets1)["mass"]
    masses2 = utils.jet_features(jets2)["mass"]

    w1s = []

    for j in range(num_batches):
        rand1 = rng.choice(len(masses1), size=num_eval_samples)
        rand2 = rng.choice(len(masses2), size=num_eval_samples)

        rand_sample1 = masses1[rand1]
        rand_sample2 = masses2[rand2]

        w1s.append(wasserstein_distance(rand_sample1, rand_sample2))

    return np.mean(w1s), np.std(w1s) if return_std else np.mean(w1s)


def w1efp(
    jets1: Union[Tensor, np.array],
    jets2: Union[Tensor, np.array],
    particle_masses: bool = False,
    efpset_args: list = [("n==", 4), ("d==", 4), ("p==", 1)],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distances between Energy Flow Polynomials (Komiske et al. 2017 https://arxiv.org/abs/1712.07124) of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.array]): Tensor or array of jets of shape ``[num_jets, num_particles, num_features]``, with features in order ``[eta, phi, pt, (optional) mass]``. If no particle masses given (``particle_masses`` should be False), they are assumed to be 0.
        jets2 (Union[Tensor, np.array]): Tensor or array of jets, of same format as ``jets1``.
        particle_masses (bool): Whether ``jets1`` and ``jets2`` have particle masses as their 4th particle features. Defaults to False.
        efpset_args (List): Args for the energyflow.efpset function to specify which EFPs to use, as defined here https://energyflow.network/docs/efp/#efpset. Defaults to the n=4, d=5, prime EFPs.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement. Defaults to 10000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        average_over_features (bool): Average over the particle features to return a single W1-P score. Defaults to True.
        return_std (bool): Return the standard deviation as well of the W1 scores over the ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[Union[float, np.array], Union[float, np.array]]:
        - **Union[float, np.array]**: if ``average_over_features`` is True, float of average W1 scores for each particle feature, first averaged over ``num_batches``,
          else array of length ``num_particle_features`` containing average W1 scores for each feature
        - **Union[float, np.array]** `(optional, only if ``return_std`` is True)`: if ``average_over_features`` is True, float of standard deviation of all W1 scores for each particle feature, first calculated over ``num_batches`` then propagated for the final average,
          else array of length ``num_particle_features`` containing standard deviation W1 scores for each feature

    """

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"
    assert (jets1.shape[2] == 3 and not particle_masses) or (jets1.shape[2] == 4 and particle_masses), "particle feature format is incorrect"
    assert (jets2.shape[2] == 3 and not particle_masses) or (jets2.shape[2] == 4 and particle_masses), "particle feature format is incorrect"

    # convert from JetNet [eta, phi, pt] format to energyflow [pt, eta, phi]
    if particle_masses:
        jets1 = jets1[:, :, [2, 0, 1, 3]]
        jets2 = jets2[:, :, [2, 0, 1, 3]]
    else:
        # pad 0 mass as the 4th feature for each particle
        jets1 = np.pad(jets1[:, :, [2, 0, 1]], ((0, 0), (0, 0), (0, 1)))
        jets2 = np.pad(jets2[:, :, [2, 0, 1]], ((0, 0), (0, 0), (0, 1)))

    efpset = EFPSet(*efpset_args, measure="hadr", beta=1, normed=None, coords="ptyphim")
    efps1 = efpset.batch_compute(jets1)
    efps2 = efpset.batch_compute(jets2)
    num_efps = efps1.shape[1]

    w1s = []

    for j in range(num_batches):
        rand1 = rng.choice(len(efps1), size=num_eval_samples)
        rand2 = rng.choice(len(efps2), size=num_eval_samples)

        rand_sample1 = efps1[rand1]
        rand_sample2 = efps2[rand2]

        w1 = [wasserstein_distance(rand_sample1[:, i], rand_sample2[:, i]) for i in range(num_efps)]
        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    if average_over_features:
        return np.mean(means), np.linalg.norm(stds) if return_std else np.mean(means)
    else:
        return means, stds if return_std else means


def cov_mmd(
    real_jets: Union[Tensor, np.array],
    gen_jets: Union[Tensor, np.array],
    num_eval_samples: int = 100,
    num_batches: int = 100,
) -> Tuple[float, float]:
    """
    Calculate coverage and MMD between real and generated jets, using the Energy Mover's Distance as the distance metric.

    Args:
        real_jets (Union[Tensor, np.array]): Tensor or array of jets, of shape ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``
        gen_jets (Union[Tensor, np.array]): tensor or array of generated jets, same format as real_jets.
        num_eval_samples (int): number of jets out of the real and gen jets each between which to evaluate COV and MMD. Defaults to 100.
        num_batches (int): number of different batches to calculate COV and MMD and average over. Defaults to 100.
        in_energyflow_format (bool): are the jets in energyflow or JetNet format. Defaults to False.

    Returns:
        Tuple[float, float]:
        - **float**: coverage, averaged over ``num_batches``.
        - **float**: MMD, averaged over ``num_batches``.

    """

    if isinstance(real_jets, Tensor):
        real_jets = real_jets.cpu().detach().numpy()

    if isinstance(gen_jets, Tensor):
        gen_jets = gen_jets.cpu().detach().numpy()

    # convert from JetNet [eta, phi, pt] format to energyflow [pt, eta, phi]
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

        # for MMD, for each gen jet find the closest real jet and average EMDs
        mmds.append(np.mean(np.min(dists, axis=0)))

        # for coverage, for each real jet find the closest gen jet and get the number of unique matchings
        covs.append(np.unique(np.argmin(dists, axis=1)).size / num_eval_samples)

    return np.mean(covs), np.mean(mmds)
