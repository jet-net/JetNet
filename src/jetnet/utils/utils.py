from typing import Dict, Union

import numpy as np
from torch import Tensor

# for calculating jet features quickly, TODO: replace with vector library when summing over axis feature is implemented
import awkward as ak
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)


def jet_features(jets: Union[Tensor, np.array]) -> Dict[str, np.array]:
    """
    Calculates jet features by summing over particle Lorentz 4-vectors.

    Args:
        jets (Union[Tensor, np.array]): Tensor or array of either a single or multiple jets, of shape either ``[num_particles, num_features]`` or ``[num_jets, num_particles, num_features]``,
          with features in order ``[eta, phi, pt, (optional) mass]``. If no particle masses given, they are assumed to be 0.

    Returns:
        Dict[str, Union[float, np.array]]: dict of float (if inputted single jet) or 1D arrays of length ``num_jets`` (if inputted multiple jets) with 'mass', 'pt', and 'eta' keys.

    """
    
    assert len(jets.shape) == 2 or len(jets.shape == 3), 'jets dimensions are incorrect'

    if len(jets.shape) == 2:
        vecs = ak.zip(
            {
                "pt": jets[:, 2:3],
                "eta": jets[:, 0:1],
                "phi": jets[:, 1:2],
                "mass": ak.full_like(jets[:, 2:3], 0) if jets.shape[1] == 3 else jets[:, 3:4],  # 0s for mass if no mass given
            },
            with_name="PtEtaPhiMLorentzVector"
        )

        sum_vecs = vecs.sum(axis=0)
    else:
        vecs = ak.zip(
            {
                "pt": jets[:, :, 2:3],
                "eta": jets[:, :, 0:1],
                "phi": jets[:, :, 1:2],
                "mass": ak.full_like(jets[:, :, 2:3], 0) if jets.shape[2] == 3 else jets[:, :, 3:4],  # 0s for mass if no mass given
            },
            with_name="PtEtaPhiMLorentzVector"
        )

        sum_vecs = vecs.sum(axis=1)

    jf = {
        "mass": np.nan_to_num(np.array(sum_vecs.mass)).squeeze(),
        "pt": np.nan_to_num(np.array(sum_vecs.pt)).squeeze(),
        "eta": np.nan_to_num(np.array(sum_vecs.eta)).squeeze(),
    }

    return jf
