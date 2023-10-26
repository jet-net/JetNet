import logging
import pathlib
import sys
import warnings
from typing import Tuple, Union

import numpy as np
import torch
from energyflow.emd import emds
from numba import njit, prange, set_num_threads
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.optimize import curve_fit
from scipy.stats import iqr, wasserstein_distance
from torch import Tensor
from torch.utils.data import DataLoader

from jetnet import utils
from jetnet.datasets import JetNet

rng = np.random.default_rng()

logger = logging.getLogger("jetnet")
logger.setLevel(logging.INFO)

# TODO: generic w1 method


def _check_get_ndarray(*arrs):
    """Checks if each input in ``arrs`` is a PyTorch tensor and, if so, converts to a numpy array"""
    ret_arrs = []
    for arr in arrs:
        if isinstance(arr, Tensor):
            ret_arrs.append(arr.cpu().detach().numpy())
        else:
            ret_arrs.append(arr)
    return ret_arrs[0] if len(ret_arrs) == 1 else ret_arrs


def _optional_tqdm(iter_obj, use_tqdm, total=None, desc=None):
    if use_tqdm:
        from tqdm import tqdm

        return tqdm(iter_obj, total=total, desc=desc)
    else:
        return iter_obj


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
        msg = (
            "fid calculation produces singular product; " "adding %s to diagonal of cov estimates"
        ) % eps
        logger.debug(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not (
            np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3)
            or np.isclose(np.trace(covmean.imag) / np.trace(covmean.real), 0, atol=1e-3)
        ):
            im_trace = np.trace(covmean.imag)
            re_trace = np.trace(covmean.real)
            warnings.warn(
                (
                    "Large imaginary components in covariance matrix while calculating "
                    f"Fréchet distance Im: {im_trace:.2f} Re: {re_trace:.2f}"
                ),
                RuntimeWarning,
            )

        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


_eval_module = sys.modules[__name__]
# for saving fpnd objects after the first loading
_eval_module.fpnd_dict = {"NUM_SAMPLES": 50_000}


def _get_fpnd_real_mu_sigma(
    dataset_name: str,
    jet_type: str,
    num_particles: int,
    num_particle_features: int,
    data_dir: str,
    device: str = "cpu",
    batch_size: int = 16,
    use_tqdm: bool = True,
):
    """
    Get and save the statistics of ParticleNet activations on real jets.
    These should already come with the library so this method should not need to be run by the user.
    """
    from .particlenet import _ParticleNet

    _eval_module_path = str(pathlib.Path(__file__).parent.resolve())
    resources_path = f"{_eval_module_path}/fpnd_resources/{dataset_name}/{num_particles}_particles"

    pnet = _ParticleNet(num_particles, num_particle_features).to(device)
    pnet.load_state_dict(torch.load(f"{resources_path}/pnet_state_dict.pt", map_location=device))
    pnet.eval()

    if dataset_name == "jetnet":
        jets = JetNet(
            jet_type,
            data_dir,
            particle_normalisation=JetNet.fpnd_norm,
            split="valid",
            split_fraction=[0.7, 0.3, 0.0],
            particle_features=["etarel", "phirel", "ptrel"],
            jet_features=None,
        ).data[: _eval_module.fpnd_dict["NUM_SAMPLES"]]
        # TODO other datasets
    else:
        raise RuntimeError("Only jetnet dataset implemented currently")

    # run inference and store activations
    jets_loaded = DataLoader(jets, batch_size)

    logger.info(f"Calculating ParticleNet activations on real jets with batch size {batch_size}")
    activations = []
    for i, jets_batch in _optional_tqdm(
        enumerate(jets_loaded), use_tqdm, total=len(jets_loaded), desc="Running ParticleNet"
    ):
        activations.append(pnet(jets_batch.to(device), ret_activations=True).cpu().detach().numpy())

    activations = np.concatenate(activations, axis=0)

    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    np.savetxt(f"{resources_path}/{jet_type}_mu.txt", mu)
    np.savetxt(f"{resources_path}/{jet_type}_sigma.txt", sigma)


def _init_fpnd_dict(
    dataset_name: str,
    jet_type: str,
    num_particles: int,
    num_particle_features: int,
    device: str = "cpu",
):
    """Load the ParticleNet model and pre-saved statistics for real jets"""
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
    _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["mu"] = np.loadtxt(
        f"{resources_path}/{jet_type}_mu.txt"
    )
    _eval_module.fpnd_dict[dataset_name][num_particles][jet_type]["sigma"] = np.loadtxt(
        f"{resources_path}/{jet_type}_sigma.txt"
    )


def fpnd(
    jets: Union[Tensor, np.ndarray],
    jet_type: str,
    dataset_name: str = "jetnet",
    device: str = None,
    batch_size: int = 16,
    use_tqdm: bool = True,
) -> float:
    """
    Calculates the Frechet ParticleNet Distance, as defined in https://arxiv.org/abs/2106.11535,
    for input ``jets`` of type ``jet_type``.

    ``jets`` are passed through our pretrained ParticleNet module and activations are compared
    with the cached activations from real jets.
    The recommended and max number of jets is 50,000.

    **torch_geometric must be installed separately for running inference with ParticleNet.**

    Currently FPND only supported for the JetNet dataset with 30 particles, but functionality for
    other datasets + ability for users to use their own version is in development.

    Args:
        jets (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape
          ``[num_jets, num_particles, num_features]`` with features in order
          ``[eta, phi, pt, (optional) mask]``
        jet_type (str): jet type, out of ``['g', 't', 'q']``.
        dataset_name (str): Dataset to use. Currently only JetNet is supported.
          Defaults to "jetnet".
        device (str): 'cpu' or 'cuda'. If not specified, defaults to cuda if available else cpu.
        batch_size (int): Batch size for ParticleNet inference. Defaults to 16.
        use_tqdm (bool): use tqdm bar while getting ParticleNet activations. Defaults to True.

    Returns:
        float: the measured FPND.

    """
    assert dataset_name == "jetnet", "Only JetNet is currently supported with FPND"

    num_particles = jets.shape[1]
    num_particle_features = jets.shape[2]

    assert (
        num_particles == 30
    ), "Currently FPND only supported for 30 particles - more functionality coming soon."
    assert (
        num_particle_features == 3
    ), "Not the right number of particle features for the JetNet dataset."

    if jets.shape[0] < _eval_module.fpnd_dict["NUM_SAMPLES"]:
        warnings.warn(
            (
                "Recommended number of jets for FPND calculation is "
                + f"{_eval_module.fpnd_dict['NUM_SAMPLES']}"
            ),
            RuntimeWarning,
        )

    if isinstance(jets, np.ndarray):
        jets = Tensor(jets)

    jets = jets.clone()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    assert device == "cuda" or device == "cpu", "Invalid device type"

    if dataset_name == "jetnet":
        JetNet.fpnd_norm(jets, inplace=True)
        # TODO other datasets

    # ParticleNet module and the real mu's and sigma's are cached in memory after the first load
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
    jets_loaded = DataLoader(jets[: _eval_module.fpnd_dict["NUM_SAMPLES"]], batch_size)

    logger.info(f"Calculating ParticleNet activations with batch size: {batch_size}")
    activations = []
    for i, jets_batch in _optional_tqdm(
        enumerate(jets_loaded), use_tqdm, total=len(jets_loaded), desc="Running ParticleNet"
    ):
        activations.append(pnet(jets_batch.to(device), ret_activations=True).cpu().detach().numpy())

    activations = np.concatenate(activations, axis=0)

    mu2 = np.mean(activations, axis=0)
    sigma2 = np.cov(activations, rowvar=False)

    fpnd = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return fpnd


def w1p(
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
    mask1: Union[Tensor, np.ndarray] = None,
    mask2: Union[Tensor, np.ndarray] = None,
    exclude_zeros: bool = True,
    num_particle_features: int = 0,
    num_eval_samples: int = 50_000,
    num_batches: int = 5,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distances between particle features of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape
          ``[num_jets, num_particles_per_jet, num_features_per_particle]``.
        jets2 (Union[Tensor, np.ndarray]): Tensor or array of jets, of same format as ``jets1``.
        mask1 (Union[Tensor, np.ndarray]): Optional tensor or array of binary particle masks, of
          shape ``[num_jets, num_particles_per_jet]`` or ``[num_jets, num_particles_per_jet, 1]``.
          If given, 0-masked particles will be excluded from w1 calculation.
        mask2 (Union[Tensor, np.ndarray]): Optional tensor or array of same format as ``masks2``.
        exclude_zeros (bool): Ignore zero-padded particles i.e.
          those whose whose feature norms are exactly 0. Defaults to True.
        num_particle_features (int): Will return W1 scores of the first
          ``num_particle_features`` particle features. If 0, will calculate for all.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement.
          Defaults to 50,000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        return_std (bool): Return the standard deviation as well of the W1 scores over the
          ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        - **Union[float, np.ndarray]**:  array of length ``num_particle_features`` containing
          average W1 scores for each feature.
        - **Union[float, np.ndarray]** `(optional, only if ``return_std`` is True)`: array of length
          ``num_particle_features`` containing standard deviation W1 scores for each feature.

    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if len(jets1) < 50_000 or len(jets2) < 50_000:
        warnings.warn("Recommended number of jets for W1 estimation is 50,000", RuntimeWarning)

    if num_particle_features <= 0:
        num_particle_features = jets1.shape[2]

    assert (
        num_particle_features <= jets1.shape[2]
    ), "more particle features requested than were inputted"
    assert (
        num_particle_features <= jets2.shape[2]
    ), "more particle features requested than were inputted"

    if mask1 is not None:
        # TODO: should be wrapped in try catch
        mask1 = mask1.reshape(jets1.shape[0], jets1.shape[1])
        mask1 = mask1.astype(bool)

    if mask2 is not None:
        # TODO: should be wrapped in try catch
        mask2 = mask2.reshape(jets2.shape[0], jets2.shape[1])
        mask2 = mask2.astype(bool)

    jets1, jets2 = _check_get_ndarray(jets1, jets2)

    if exclude_zeros:
        zeros1 = np.linalg.norm(jets1[:, :, :num_particle_features], axis=2) == 0
        mask1 = ~zeros1 if mask1 is None else mask1 * ~zeros1

        zeros2 = np.linalg.norm(jets2[:, :, :num_particle_features], axis=2) == 0
        mask2 = ~zeros2 if mask2 is None else mask2 * ~zeros2

    w1s = []

    for j in range(num_batches):
        rand1 = rng.choice(len(jets1), size=num_eval_samples)
        rand2 = rng.choice(len(jets2), size=num_eval_samples)

        rand_sample1 = jets1[rand1]
        rand_sample2 = jets2[rand2]

        if mask1 is not None:
            parts1 = rand_sample1[:, :, :num_particle_features][mask1[rand1]]
        else:
            parts1 = rand_sample1[:, :, :num_particle_features].reshape(-1, num_particle_features)

        if mask2 is not None:
            parts2 = rand_sample2[:, :, :num_particle_features][mask2[rand2]]
        else:
            parts2 = rand_sample2[:, :, :num_particle_features].reshape(-1, num_particle_features)

        if parts1.shape[0] == 0 or parts2.shape[0] == 0:
            w1 = [np.inf, np.inf, np.inf]
        else:
            w1 = [
                wasserstein_distance(parts1[:, i], parts2[:, i])
                for i in range(num_particle_features)
            ]

        w1s.append(w1)

    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)

    return (means, stds) if return_std else means


def w1m(
    jets1: Union[Tensor, np.ndarray],
    jets2: Union[Tensor, np.ndarray],
    num_eval_samples: int = 50_000,
    num_batches: int = 5,
    return_std: bool = True,
):
    """
    Get 1-Wasserstein distance between masses of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape
          ``[num_jets, num_particles, num_features]`` with features in order
          ``[eta, phi, pt, (optional) mass]``
        jets2 (Union[Tensor, np.ndarray]): Tensor or array of jets, of same format as ``jets1``.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement.
          Defaults to 50,000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        return_std (bool): Return the standard deviation as well of the W1 scores over the
          ``num_batches`` batches. Defaults to True.

    Returns:
        Tuple[float, float]:
        - **float**: W1 mass score, averaged over ``num_batches``.
        - **float** `(optional, only if ```return_std``` is True)`: standard deviation of W1 mass
          scores over ``num_batches``.

    """
    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"

    if len(jets1) < 50_000 or len(jets2) < 50_000:
        warnings.warn("Recommended number of jets for W1 estimation is 50,000", RuntimeWarning)

    jets1, jets2 = _check_get_ndarray(jets1, jets2)

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
    num_eval_samples: int = 50_000,
    num_batches: int = 5,
    return_std: bool = True,
    efp_jobs: int = None,
):
    """
    Get 1-Wasserstein distances between Energy Flow Polynomials
    (Komiske et al. 2017 https://arxiv.org/abs/1712.07124) of ``jets1`` and ``jets2``.

    Args:
        jets1 (Union[Tensor, np.ndarray]): Tensor or array of jets of shape
          ``[num_jets, num_particles, num_features]``, with features in order
          ``[eta, phi, pt, (optional) mass]``. If no particle masses given
          (``particle_masses`` should be False), they are assumed to be 0.
        jets2 (Union[Tensor, np.ndarray]): Tensor or array of jets, of same format as ``jets1``.
        use_particle_masses (bool): Whether ``jets1`` and ``jets2`` have particle masses as their
          4th particle features. Defaults to False.
        efpset_args (List): Args for the energyflow.efpset function to specify which EFPs to use,
          as defined here https://energyflow.network/docs/efp/#efpset.
          Defaults to the n=4, d=5, prime EFPs.
        num_eval_samples (int): Number of jets out of the total to use for W1 measurement.
          Defaults to 50,000.
        num_batches (int): Number of different batches to average W1 scores over. Defaults to 5.
        average_over_efps (bool): Average over the EFPs to return a single W1-EFP score.
          Defaults to True.
        return_std (bool): Return the standard deviation as well of the W1 scores over the
          ``num_batches`` batches. Defaults to True.
        efp_jobs (int): number of jobs to use for energyflow's EFP batch computation.
          None means as many processes as there are CPUs.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
        - **np.ndarray**:  array of average W1 scores for each EFP.
        - **np.ndarray** `(optional, only if return_std is True)`: array of std of W1 scores for
          each feature.

    """
    if len(jets1) < 50_000 or len(jets2) < 50_000:
        warnings.warn("Recommended number of jets for W1 estimation is 50,000", RuntimeWarning)

    jets1, jets2 = _check_get_ndarray(jets1, jets2)

    assert len(jets1.shape) == 3 and len(jets2.shape) == 3, "input jets format is incorrect"
    assert (jets1.shape[2] - int(use_particle_masses) >= 3) and (
        jets1.shape[2] - int(use_particle_masses) >= 3
    ), "particle feature format is incorrect"

    efps1 = utils.efps(
        jets1, use_particle_masses=use_particle_masses, efpset_args=efpset_args, efp_jobs=efp_jobs
    )
    efps2 = utils.efps(
        jets2, use_particle_masses=use_particle_masses, efpset_args=efpset_args, efp_jobs=efp_jobs
    )
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

    return (means, stds) if return_std else means


def cov_mmd(
    real_jets: Union[Tensor, np.ndarray],
    gen_jets: Union[Tensor, np.ndarray],
    num_eval_samples: int = 100,
    num_batches: int = 10,
    use_tqdm: bool = True,
) -> Tuple[float, float]:
    """
    Calculate coverage and MMD between real and generated jets,
    using the Energy Mover's Distance as the distance metric.

    Args:
        real_jets (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape
          ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``
        gen_jets (Union[Tensor, np.ndarray]): tensor or array of generated jets,
          same format as real_jets.
        num_eval_samples (int): number of jets out of the real and gen jets each between which to
          evaluate COV and MMD. Defaults to 100.
        num_batches (int): number of different batches to calculate COV and MMD and average over.
          Defaults to 100.
        use_tqdm (bool): use tqdm bar while calculating over ``num_batches`` batches.
          Defaults to True.

    Returns:
        Tuple[float, float]:
        - **float**: coverage, averaged over ``num_batches``.
        - **float**: MMD, averaged over ``num_batches``.

    """
    assert len(real_jets.shape) == 3 and len(gen_jets.shape) == 3, "input jets format is incorrect"
    assert (real_jets.shape[2] >= 3) and (
        gen_jets.shape[2] >= 3
    ), "particle feature format is incorrect"

    real_jets, gen_jets = _check_get_ndarray(real_jets, gen_jets)

    assert np.all(real_jets[:, :, 2] >= 0) and np.all(
        gen_jets[:, :, 2] >= 0
    ), "particle pTs must all be >= 0 for EMD calculation"

    # convert from JetNet [eta, phi, pt] format to energyflow [pt, eta, phi]
    real_jets = real_jets[:, :, [2, 0, 1]]
    gen_jets = gen_jets[:, :, [2, 0, 1]]

    covs = []
    mmds = []

    for j in _optional_tqdm(
        range(num_batches), use_tqdm, desc=f"Calculating cov and mmd over {num_batches} batches"
    ):
        real_rand = rng.choice(len(real_jets), size=num_eval_samples)
        gen_rand = rng.choice(len(gen_jets), size=num_eval_samples)

        real_rand_sample = real_jets[real_rand]
        gen_rand_sample = gen_jets[gen_rand]

        # 2D array of emds, with shape (len(gen_rand_sample), len(real_rand_sample))
        dists = emds(gen_rand_sample, real_rand_sample)

        # for MMD, for each gen jet find the closest real jet and average EMDs
        mmds.append(np.mean(np.min(dists, axis=0)))

        # for coverage, for each real jet find the closest gen jet
        # and get the number of unique matchings
        covs.append(np.unique(np.argmin(dists, axis=1)).size / num_eval_samples)

    return np.mean(covs), np.mean(mmds)


def get_fpd_kpd_jet_features(jets: Union[Tensor, np.ndarray], efp_jobs: int = None) -> np.ndarray:
    """Get recommended jet features (36 EFPs) for the FPD and KPD metrics from an input sample of
    jets.

    Args:
        jets (Union[Tensor, np.ndarray]): Tensor or array of jets, of shape
          ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``.
        efp_jobs (int, optional): number of jobs to use for energyflow's EFP batch computation.
          None means as many processes as there are CPUs.

    Returns:
        np.ndarray: array of EFPs of shape ``[num_jets, 36]``.
    """
    jets = _check_get_ndarray(jets)
    return utils.efps(jets, efpset_args=[("d<=", 4)], efp_jobs=efp_jobs)


def _normalise_features(X: ArrayLike, Y: ArrayLike = None) -> ArrayLike:
    maxes = np.max(np.abs(X), axis=0)
    maxes[maxes == 0] = 1  # don't normalise in case of features which are just 0

    return (X / maxes, Y / maxes) if Y is not None else X / maxes


def _linear(x, intercept, slope):
    return intercept + slope * x


# based on https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
def fpd(
    real_features: Union[Tensor, np.ndarray],
    gen_features: Union[Tensor, np.ndarray],
    min_samples: int = 20_000,
    max_samples: int = 50_000,
    num_batches: int = 20,
    num_points: int = 10,
    normalise: bool = True,
    seed: int = 42,
) -> Tuple[float, float]:
    """Calculates the value and error of the Fréchet physics distance (FPD) between a set of real
    and generated features, as defined in https://arxiv.org/abs/2211.10295.

    It is recommended to use input sample sizes of at least 50,000, and the default values for other
    input parameters for consistency with other measurements.

    Similarly, for jets, it is recommended to use the set of EFPs as provided by the
    ``get_fpd_kpd_jet_features`` method.

    Args:
        real_features (Union[Tensor, np.ndarray]): set of real features of shape
          ``[num_samples, num_features]``.
        gen_features (Union[Tensor, np.ndarray]): set of generated features of shape
        ``[num_samples, num_features]``.
        min_samples (int, optional): min batch size to measure FPD for. Defaults to 20,000.
        max_samples (int, optional): max batch size to measure FPD for. Defaults to 50,000.
        num_batches (int, optional): # of batches to average over for each batch size.
          Defaults to 20.
        num_points (int, optional): # of points to sample between the min and max samples.
          Defaults to 10.
        normalise (bool, optional): normalise the individual features over the full sample to have
          the same scaling. Defaults to True.
        seed (int, optional): random seed. Defaults to 42.

    Returns:
        Tuple[float, float]: value and error of FPD.
    """
    if len(real_features) < 50_000 or len(gen_features) < 50_000:
        warnings.warn("Recommended number of samples for FPD estimation is 50,000", RuntimeWarning)

    real_features, gen_features = _check_get_ndarray(real_features, gen_features)

    if normalise:
        X, Y = _normalise_features(real_features, gen_features)

    # regular intervals in 1/N
    batches = (1 / np.linspace(1.0 / min_samples, 1.0 / max_samples, num_points)).astype("int32")
    np.random.seed(seed)
    vals = []

    for i, batch_size in enumerate(batches):
        val_points = []  # values per batch size
        for _ in range(num_batches):
            rand1 = np.random.choice(len(X), size=batch_size)
            rand2 = np.random.choice(len(Y), size=batch_size)

            rand_sample1 = X[rand1]
            rand_sample2 = Y[rand2]

            mu1 = np.mean(rand_sample1, axis=0)
            sigma1 = np.cov(rand_sample1, rowvar=False)
            mu2 = np.mean(rand_sample2, axis=0)
            sigma2 = np.cov(rand_sample2, rowvar=False)

            val = _calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
            val_points.append(val)

        vals.append(np.mean(val_points))

    params, covs = curve_fit(_linear, 1 / batches, vals, bounds=([0, 0], [np.inf, np.inf]))
    return (params[0], np.sqrt(np.diag(covs)[0]))  # y-intercept, err on y-intercept


@njit
def _poly_kernel_pairwise(X: ArrayLike, Y: ArrayLike, degree: int) -> np.ndarray:
    """Pairwise polynomial kernel of degree ``degree`` between X and Y"""
    gamma = 1.0 / X.shape[-1]
    return (X @ Y.T * gamma + 1.0) ** degree


@njit
def _mmd_quadratic_unbiased(XX: ArrayLike, YY: ArrayLike, XY: ArrayLike):
    """Calculate quadratic estimate for MMD given pairwise distances between X and Y"""
    m, n = XX.shape[0], YY.shape[0]
    # subtract diagonal 1s
    return (
        (XX.sum() - np.trace(XX)) / (m * (m - 1))
        + (YY.sum() - np.trace(YY)) / (n * (n - 1))
        - 2 * np.mean(XY)
    )


@njit
def _mmd_poly_quadratic_unbiased(X: ArrayLike, Y: ArrayLike, degree: int = 4) -> float:
    """Calculate quadratic estimate for MMD with a polynomial kernel of degree ``degree``"""
    XX = _poly_kernel_pairwise(X, X, degree=degree)
    YY = _poly_kernel_pairwise(Y, Y, degree=degree)
    XY = _poly_kernel_pairwise(X, Y, degree=degree)
    return _mmd_quadratic_unbiased(XX, YY, XY)


@njit(parallel=True)
def _kpd_batches_parallel(X, Y, num_batches, batch_size, seed):
    vals_point = np.zeros(num_batches, dtype=np.float64)
    for i in prange(num_batches):
        np.random.seed(seed + i * 1000)  # in case of multi-threading
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = _mmd_poly_quadratic_unbiased(rand_sample1, rand_sample2, degree=4)
        vals_point[i] = val

    return vals_point


def _kpd_batches(X, Y, num_batches, batch_size, seed):
    vals_point = []
    for i in range(num_batches):
        np.random.seed(seed + i * 1_000)
        rand1 = np.random.choice(len(X), size=batch_size)
        rand2 = np.random.choice(len(Y), size=batch_size)

        rand_sample1 = X[rand1]
        rand_sample2 = Y[rand2]

        val = _mmd_poly_quadratic_unbiased(rand_sample1, rand_sample2)
        vals_point.append(val)

    return vals_point


def kpd(
    real_features: Union[Tensor, np.ndarray],
    gen_features: Union[Tensor, np.ndarray],
    num_batches: int = 10,
    batch_size: int = 5_000,
    normalise: bool = True,
    seed: int = 42,
    num_threads: int = None,
) -> Tuple[float, float]:
    """Calculates the median and error of the kernel physics distance (KPD) between a set of real
    and generated features, as defined in https://arxiv.org/abs/2211.10295.

    It is recommended to use input sample sizes of at least 50,000, and the default values for other
    input parameters for consistency with other measurements.

    Similarly, for jets, it is recommended to use the set of EFPs as provided by the
    ``get_fpd_kpd_jet_features`` method.

    Args:
        real_features (Union[Tensor, np.ndarray]): set of real features of shape
          ``[num_samples, num_features]``.
        gen_features (Union[Tensor, np.ndarray]): set of generated features of shape
          ``[num_samples, num_features]``.
        num_batches (int, optional): number of batches to average over. Defaults to 10.
        batch_size (int, optional): size of each batch for which MMD is measured. Defaults to 5,000.
        normalise (bool, optional): normalise the individual features over the full sample to have
          the same scaling. Defaults to True.
        seed (int, optional): random seed. Defaults to 42.
        num_threads (int, optional): parallelize KPD through numba using this many threads. 0 means
          numba's default number of threads, based on # of cores available. Defaults to None, i.e.
          no parallelization.

    Returns:
        Tuple[float, float]: median and error of KPD.
    """
    real_features, gen_features = _check_get_ndarray(real_features, gen_features)

    if normalise:
        X, Y = _normalise_features(real_features, gen_features)

    if num_threads is None:
        vals_point = _kpd_batches(X, Y, num_batches, batch_size, seed)
    else:
        if num_threads > 0:
            set_num_threads(num_threads)

        vals_point = _kpd_batches_parallel(X, Y, num_batches, batch_size, seed)

    # median, error = half of 16 - 84 IQR
    return (np.median(vals_point), iqr(vals_point, rng=(16.275, 83.725)) / 2)
