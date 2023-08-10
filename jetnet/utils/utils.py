from __future__ import annotations  # for ArrayLike type in docs

from typing import Dict, Tuple, Union

# for calculating jet features quickly,
# TODO: replace with vector library when summing over axis feature is implemented
import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
from energyflow import EFPSet
from numpy.typing import ArrayLike

ak.behavior.update(vector.behavior)


def jet_features(jets: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculates jet features by summing over particle Lorentz 4-vectors.

    Args:
        jets (np.ndarray): array of either a single or multiple jets, of shape either
          ``[num_particles, num_features]`` or ``[num_jets, num_particles, num_features]``,
          with features in order ``[eta, phi, pt, (optional) mass]``. If no particle masses given,
          they are assumed to be 0.

    Returns:
        Dict[str, Union[float, np.ndarray]]:
          dict of float (if inputted single jet) or
          1D arrays of length ``num_jets`` (if inputted multiple jets)
          with 'mass', 'pt', and 'eta' keys.

    """

    assert len(jets.shape) == 2 or len(jets.shape) == 3, "jets dimensions are incorrect"
    assert jets.shape[-1] >= 3, "missing particle features"

    if len(jets.shape) == 2:
        vecs = ak.zip(
            {
                "pt": jets[:, 2:3],
                "eta": jets[:, 0:1],
                "phi": jets[:, 1:2],
                # 0s for mass if no mass given
                "mass": ak.full_like(jets[:, 2:3], 0) if jets.shape[1] == 3 else jets[:, 3:4],
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        sum_vecs = vecs.sum(axis=0)
    else:
        vecs = ak.zip(
            {
                "pt": jets[:, :, 2:3],
                "eta": jets[:, :, 0:1],
                "phi": jets[:, :, 1:2],
                # 0s for mass if no mass given
                "mass": ak.full_like(jets[:, :, 2:3], 0) if jets.shape[2] == 3 else jets[:, :, 3:4],
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        sum_vecs = vecs.sum(axis=1)

    jf = {
        "mass": np.nan_to_num(np.array(sum_vecs.mass)).squeeze(),
        "pt": np.nan_to_num(np.array(sum_vecs.pt)).squeeze(),
        "eta": np.nan_to_num(np.array(sum_vecs.eta)).squeeze(),
    }

    return jf


def efps(
    jets: np.ndarray,
    use_particle_masses: bool = False,
    efpset_args: list = [("n==", 4), ("d==", 4), ("p==", 1)],
    efp_jobs: int = None,
) -> np.ndarray:
    """
    Utility for calculating EFPs for jets in JetNet format using the energyflow library.

    Args:
        jets (np.ndarray): array of either a single or multiple jets, of shape either
          ``[num_particles, num_features]`` or ``[num_jets, num_particles, num_features]``,
          with features in order ``[eta, phi, pt, (optional) mass]``. If no particle masses given,
          they are assumed to be 0.
        efpset_args (List): Args for the energyflow.efpset function to specify which EFPs to use,
          as defined here https://energyflow.network/docs/efp/#efpset.
          Defaults to the n=4, d=5, prime EFPs.
        efp_jobs (int): number of jobs to use for energyflow's EFP batch computation.
          None means as many processes as there are CPUs.

    Returns:
        np.ndarray:
          1D (if inputted single jet) or 2D array of shape ``[num_jets, num_efps]`` of EFPs per jet

    """

    assert len(jets.shape) == 2 or len(jets.shape) == 3, "jets dimensions are incorrect"
    assert jets.shape[-1] - int(use_particle_masses) >= 3, "particle feature format is incorrect"

    efpset = EFPSet(*efpset_args, measure="hadr", beta=1, normed=None, coords="ptyphim")

    if len(jets.shape) == 2:
        # convert to energyflow format
        jets = jets[:, [2, 0, 1]] if not use_particle_masses else jets[:, [2, 0, 1, 3]]
        efps = efpset.compute(jets)
    else:
        # convert to energyflow format
        jets = jets[:, :, [2, 0, 1]] if not use_particle_masses else jets[:, :, [2, 0, 1, 3]]
        efps = efpset.batch_compute(jets, efp_jobs)

    return efps


def to_image(
    jets: np.ndarray, im_size: int, mask: np.ndarray = None, maxR: float = 1.0
) -> np.ndarray:
    """
    Convert jet(s) into 2D ``im_size`` x ``im_size`` or  3D ``num_jets`` x ``im_size`` x ``im_size``
    image arrays.

    Args:
        jets (np.ndarray): array of jet(s) of shape ``[num_particles, num_features]`` or
          ``[num_jets, num_particles, num_features]`` with features in order ``[eta, phi, pt]``.
        im_size (int): number of pixels per row and column.
        mask (np.ndarray): optional binary array of masks of shape ``[num_particles]`` or
          ``[num_jets, num_particles]``.
        maxR (float): max radius of the jet. Defaults to 1.0.

    Returns:
        np.ndarray: 2D or 3D array of shape ``[im_size, im_size]`` or
        ``[num_jets, im_size, im_size]``.

    """
    assert len(jets.shape) == 2 or len(jets.shape) == 3, "jets dimensions are incorrect"
    assert jets.shape[-1] >= 3, "particle feature format is incorrect"

    eta = jets[..., 0]
    phi = jets[..., 1]
    pt = jets[..., 2]
    if len(jets.shape) == 2:
        num_jets = 1
    else:
        num_jets = jets.shape[0]

    if mask is not None:
        assert len(mask.shape) == 1 or len(mask.shape) == 2, "mask shape incorrect"
        assert mask.shape == jets.shape[:-1], "mask shape and jets shape do not agree"
        mask = mask.astype(int)
        pt *= mask

    jet_images = np.zeros((num_jets, im_size, im_size))

    for i_jet in range(num_jets):
        hist_2d, _, _ = np.histogram2d(
            eta[i_jet] if num_jets > 1 else eta,
            phi[i_jet] if num_jets > 1 else phi,
            bins=[im_size, im_size],
            range=[[-maxR, maxR], [-maxR, maxR]],
            weights=pt[i_jet] if num_jets > 1 else pt,
        )
        jet_images[i_jet] = hist_2d

    if num_jets == 1:
        jet_images = jet_images[0]

    return jet_images


def gen_jet_corrections(
    jets: ArrayLike,
    ret_mask_separate: bool = True,
    zero_mask_particles: bool = True,
    zero_neg_pt: bool = True,
    pt_index: int = 2,
) -> Union[ArrayLike, Tuple[ArrayLike, ArrayLike]]:
    """
    Zero's masked particles and negative pTs.

    Args:
        jets (ArrayLike): jets to recorrect.
        ret_mask_separate (bool, optional): return the jet and mask separately. Defaults to True.
        zero_mask_particles (bool, optional): set features of zero-masked particles to 0. Defaults
            to True.
        zero_neg_pt (bool, optional): set pT to 0 for particles with negative pt. Defaults to True.
        pt_index (int, optional): index of the pT feature. Defaults to 2.

    Returns:
        Jets of same type as input, of shape
        ``[num_jets, num_particles, num_features (including mask)]`` if ``ret_mask_separate``
        is False, else a tuple with a tensor/array of shape
        ``[num_jets, num_particles, num_features (excluding mask)]`` and another binary mask
        tensor/array of shape ``[num_jets, num_particles, 1]``.
    """

    use_mask = ret_mask_separate or zero_mask_particles

    mask = jets[:, :, -1] >= 0.5 if use_mask else None

    if zero_mask_particles and use_mask:
        jets[~mask] = 0

    if zero_neg_pt:
        jets[:, :, pt_index][jets[:, :, pt_index] < 0] = 0

    return (jets[:, :, :-1], mask) if ret_mask_separate else jets
