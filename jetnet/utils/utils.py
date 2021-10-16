from typing import Dict, Union

import numpy as np

# for calculating jet features quickly, TODO: replace with vector library when summing over axis feature is implemented
import awkward as ak
from coffea.nanoevents.methods import vector

from energyflow import EFPSet

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
        Dict[str, Union[float, np.ndarray]]: dict of float (if inputted single jet) or
          1D arrays of length ``num_jets`` (if inputted multiple jets) with 'mass', 'pt', and 'eta' keys.

    """

    assert len(jets.shape) == 2 or len(jets.shape) == 3, "jets dimensions are incorrect"

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
        np.ndarray: 1D (if inputted single jet) or 2D array of shape [num_jets, num_efps] of EFPs per jet

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
    jet: np.ndarray, im_size: int, mask: np.ndarray = None, maxR: float = 1.0
) -> np.ndarray:
    """
    Convert a single jet into a 2D ``im_size`` x ``im_size`` array.

    Args:
        jet (np.ndarray): array of a single jet of shape ``[num_particles, num_features]``
          with features in order ``[eta, phi, pt]``.
        im_size (int): number of pixels per row and column.
        mask (np.ndarray): optional binary array of masks of shape ``[num_particles]``.
        maxR (float): max radius of the jet. Defaults to 1.0.

    Returns:
        np.ndarray: 2D array of shape ``[im_size, im_size]``.

    """
    assert len(jet.shape) == 2, "jets dimensions are incorrect"
    assert jet.shape[-1] >= 3, "particle feature format is incorrect"

    bins = np.linspace(-maxR, maxR, im_size + 1)
    binned_eta = np.digitize(jet[:, 0], bins) - 1
    binned_phi = np.digitize(jet[:, 1], bins) - 1
    pt = jet[:, 2]

    if mask is not None:
        assert len(mask.shape) == 1 and mask.shape[0] == jet.shape[0], "mask format incorrect"
        mask = mask.astype(int)
        pt *= mask

    jet_image = np.zeros((im_size, im_size))

    for eta, phi, pt in zip(binned_eta, binned_phi, pt):
        if eta >= 0 and eta < im_size and phi >= 0 and phi < im_size:
            jet_image[phi, eta] += pt

    return jet_image
