# energyflow needs to be imported before pytorch because of https://github.com/pkomiske/EnergyFlow/issues/24
from energyflow.emd import emds

import logging
import warnings
from typing import Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from jetnet.datasets import JetNet
from jetnet import utils

from scipy import linalg
from scipy.stats import wasserstein_distance

# from .particlenet import ParticleNet

from tqdm import tqdm

import pathlib
import sys


rng = np.random.default_rng()


# TODOs:
# - Functionality for adding FPND for new datasets
# - Cartesian coordinates


# from https://github.com/mseitzer/pytorch-fid
def _calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; " "adding %s to diagonal of cov estimates") % eps
        logging.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


_eval_module = sys.modules[__name__]
# for saving fpnd objects after the first loading
_eval_module.fpnd_dict = {"NUM_SAMPLES": 50000}


def _init_fpnd_dict(dataset_name: str, jet_type: str, num_particles: int, num_particle_features: int, device: str = "cpu"):
    try:
        from .particlenet import _ParticleNet
    except ModuleNotFoundError:
        print("torch_geometric needs to be installed for FPND")
        raise

    if dataset_name not in _eval_module.fpnd_dict:
        _eval_module.fpnd_dict[dataset_name] = {}

    if num_particles not in _eval_module.fpnd_dict[dataset_name]:
        _eval_module.fpnd_dict[dataset_name][num_particles] = {}

    if jet_type not in _eval_module.fpnd_dict[dataset_name][num_particles]:
        _eval_module.fpnd_dict[dataset_name][num_particles][jet_type] = {}

    _eval_module_path = str(pathlib.Path(__file__).parent.resolve())
    resources_path = f"{_eval_module_path}/fpnd_resources/{dataset_name}/{num_particles}_particles"

    pnet = _ParticleNet(num_particles, num_particle_features)
    pnet.load_state_dict(torch.load(f"{resources_path}/pnet_state_dict.pt", map_location=device))

    _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["pnet"] = pnet
    _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["mu"] = np.loadtxt(f"{resources_path}/{jet_type}_mu.txt")
    _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["sigma"] = np.loadtxt(f"{resources_path}/{jet_type}_sigma.txt")


# TODO !!! check gen jets are not in place normalized !!!
def fpnd(jets: Union[Tensor, np.ndarray], jet_type: str, dataset_name: str = "JetNet", device: str = None, batch_size: int = 16) -> float:
    """
    Calculates the Frechet ParticleNet Distance, as defined in https://arxiv.org/abs/2106.11535, for input ``jets`` of type ``jet_type``.

    ``jets`` are passed through our pretrained ParticleNet module and activations are compared with the cached activations from real jets.
    The recommended and max number of jets is 50,000

    **torch_geometric must be installed separately for running inference with ParticleNet**

    Currently FPND only supported for the JetNet dataset with 30 particles,
    but functionality for other datasets + ability for users to use their own version is in development.

    Args:
        jets (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt, (optional) mask]``
        jet_type (str): jet type, out of ``['g', 't', 'q']``.
        dataset_name (str): Dataset to use. Currently only JetNet is supported. Defaults to "JetNet".
        device (str): 'cpu' or 'cuda'. If not specified, defaults to cuda if available else cpu.
        batch_size (int): Batch size for ParticleNet inference. Defaults to 16.

    Returns:
        float: the measured FPND.

    """
    assert dataset_name == "JetNet", "Only JetNet is currently supported with FPND"

    num_particles = jets.shape[1]
    num_particle_features = jets.shape[2]

    assert num_particles == 30, "Currently FPND only supported for 30 particles - more functionality coming soon."
    assert num_particle_features == 3, "Not the right number of particle features for the JetNet dataset."

    if jets.shape[0] < _eval_module.fpnd_dict["NUM_SAMPLES"]:
        warnings.warn(f"Recommended number of jets for FPND calculation is {_eval_module.fpnd_dict['NUM_SAMPLES']}", RuntimeWarning)

    if isinstance(jets, np.ndarray):
        jets = Tensor(jets)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert device == "cuda" or device == "cpu", "Invalid device type"

    if dataset_name == "JetNet":
        JetNet.normalize_features(jets, fpnd=True)
        # TODO other datasets

    # if use_mask:
    #     # features for all masked paricles are set to 0 and mask feature is removed
    #     mask = jets[:, :, -1:] > 0
    #     jets = (jets * mask)[:, :, :-1]

    # ParticleNet module and the real mu's and sigma's are only loaded once
    if (
        dataset_name not in _eval_module.fpnd_dict
        or num_particles not in _eval_module.fpnd_dict[dataset_name]
        or jet_type not in _eval_module.fpnd_dict[dataset_name][num_particles]
    ):
        _init_fpnd_dict(dataset_name, jet_type, num_particles, num_particle_features, device)

    pnet = _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["pnet"].to(device)
    pnet.eval()

    mu1 = _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["mu"]
    sigma1 = _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["sigma"]

    # run inference and store activations
    jets_loaded = DataLoader(jets[:, :, : _eval_module.fpnd_dict["NUM_SAMPLES"]], batch_size)

    logging.info(f"Calculating ParticleNet inferences with {batch_size = }")
    activations = []
    for i, jets_batch in tqdm(enumerate(jets_loaded), total=len(jets_loaded)):
        activations.append(pnet(jets.to(device), ret_activations=True).cpu().detach().numpy())

    activations = np.array(activations)
    activations = activations.reshape(-1, activations.shape[-1])  # remove batch dimension

    mu2 = np.mean(activations, axis=0)
    sigma2 = np.cov(activations, rowvar=False)

    fpnd = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fpnd


def w1p(
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
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
        jets1 (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape ``[num_jets, num_particles_per_jet, num_features_per_particle]`` with an optional last binary mask feature per particle.
        jets2 (Union[Tensor, np.ndarray]): Tensor or array of jets, of same format as ``jets1``.
        use_mask (bool): Use the last binary mask feature to ignore 0-masked particles. Defaults to True.
        num_particle_features (int): Will return W1 scores of the first ``num_particle_features`` particle features. If 0, will calculate for all,
          excluding the optional mask feature if ``use_mask`` is True. Defaults to 0.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement. Defaults to 10000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        average_over_features (bool): Average over the particle features to return a single W1-P score. Defaults to True.
        return_std (bool): Return the standard deviation as well of the W1 scores over the ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        - **Union[float, np.ndarray]**: if ``average_over_features`` is True, float of average W1 scores for each particle feature, first averaged over ``num_batches``,
          else array of length ``num_particle_features`` containing average W1 scores for each feature
        - **Union[float, np.ndarray]** `(optional, only if ``return_std`` is True)`: if ``average_over_features`` is True, float of standard deviation of all W1 scores for each particle feature,
          first calculated over ``num_batches`` then propagated for the final average, else array of length ``num_particle_features`` containing standard deviation W1 scores for each feature

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
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
    use_particle_masses: bool = False,
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distance between masses of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt, (optional) mass]``
        jets2 (Union[Tensor, np.ndarray]): Tensor or array of jets, of same format as ``jets1``.
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
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
    use_particle_masses: bool = False,
    efpset_args: list = [("n==", 4), ("d==", 4), ("p==", 1)],
    num_eval_samples: int = 10000,
    num_batches: int = 5,
    average_over_features: bool = True,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distances between Energy Flow Polynomials (Komiske et al. 2017 https://arxiv.org/abs/1712.07124) of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.ndarray]): Tensor or array of jets of shape ``[num_jets, num_particles, num_features]``, with features in order ``[eta, phi, pt, (optional) mass]``.
          If no particle masses given (``particle_masses`` should be False), they are assumed to be 0.
        jets2 (Union[Tensor, np.ndarray]): Tensor or array of jets, of same format as ``jets1``.
        use_particle_masses (bool): Whether ``jets1`` and ``jets2`` have particle masses as their 4th particle features. Defaults to False.
        efpset_args (List): Args for the energyflow.efpset function to specify which EFPs to use, as defined here https://energyflow.network/docs/efp/#efpset.
          Defaults to the n=4, d=5, prime EFPs.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement. Defaults to 10000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        average_over_features (bool): Average over the particle features to return a single W1-P score. Defaults to True.
        return_std (bool): Return the standard deviation as well of the W1 scores over the ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        - **Union[float, np.ndarray]**: if ``average_over_features`` is True, float of average W1 scores for each particle feature, first averaged over ``num_batches``,
          else array of length ``num_particle_features`` containing average W1 scores for each feature
        - **Union[float, np.ndarray]** `(optional, only if ``return_std`` is True)`: if ``average_over_features`` is True, float of standard deviation of all W1 scores for each particle feature,
          first calculated over ``num_batches`` then propagated for the final average, else array of length ``num_particle_features`` containing standard deviation W1 scores for each feature

    """

    if isinstance(jets1, Tensor):
        jets1 = jets1.cpu().detach().numpy()

    if isinstance(jets2, Tensor):
        jets2 = jets2.cpu().detach().numpy()

    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"
    assert (jets1.shape[2] - int(use_particle_masses) >= 3) and (
        jets1.shape[2] - int(use_particle_masses) >= 3
    ), "particle feature format is incorrect"

    efps1 = utils.efps(jets1, use_particle_masses=use_particle_masses, efpset_args=efpset_args)
    efps2 = utils.efps(jets2, use_particle_masses=use_particle_masses, efpset_args=efpset_args)
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
    real_jets: Union[Tensor, np.ndarray],
    gen_jets: Union[Tensor, np.ndarray],
    num_eval_samples: int = 100,
    num_batches: int = 100,
) -> Tuple[float, float]:
    """
    Calculate coverage and MMD between real and generated jets, using the Energy Mover's Distance as the distance metric.

    Args:
        real_jets (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``
        gen_jets (Union[Tensor, np.ndarray]): tensor or array of generated jets, same format as real_jets.
        num_eval_samples (int): number of jets out of the real and gen jets each between which to evaluate COV and MMD. Defaults to 100.
        num_batches (int): number of different batches to calculate COV and MMD and average over. Defaults to 100.
        in_energyflow_format (bool): are the jets in energyflow or JetNet format. Defaults to False.

    Returns:
        Tuple[float, float]:
        - **float**: coverage, averaged over ``num_batches``.
        - **float**: MMD, averaged over ``num_batches``.

    """
    assert len(real_jets.shape) == 3 and len(gen_jets.shape) == 3, "input jets format is incorrect"
    assert (real_jets.shape[2] >= 3) and (gen_jets.shape[2] >= 3), "particle feature format is incorrect"

    if isinstance(real_jets, Tensor):
        real_jets = real_jets.cpu().detach().numpy()

    if isinstance(gen_jets, Tensor):
        gen_jets = gen_jets.cpu().detach().numpy()

    assert np.all(real_jets[:, :, 2] >= 0) and np.all(gen_jets[:, :, 2] >= 0), "particle pTs must all be >= 0 for EMD calculation"

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
