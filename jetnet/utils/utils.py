from typing import Dict, Union

import numpy as np
import torch

# for calculating jet features quickly,
# TODO: replace with vector library when summing over axis feature is implemented
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
        Dict[str, Union[float, np.ndarray]]: dict of float (if inputted single jet) or \
          1D arrays of length ``num_jets`` (if inputted multiple jets) \
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
        np.ndarray: 1D (if inputted single jet) or 2D array of shape ``[num_jets, num_efps]`` of \
          EFPs per jet

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


def get_polar(
    jet: np.ndarray or torch.Tensor,
    eps: float = 1e-12,
    return_4vec: bool = False
) -> np.ndarray or torch.Tensor:
    """
    Convert 3- or 4-vector features in Cartesian coordiantes to polar coordinates:
    :math:`(p_x, p_y, p_z)` or :math:`(|\mathbf{p}|, p_x, p_y, p_z)`
    :math:`\longrightarrow`
    :math:`(\eta, \phi, p_\mathrm{T})` or :math:`(|\mathbf{p}|, \eta, \phi, \mathrm{p}_T)`.


    Args:
        jet (np.ndarray or torch.Tensor): array or tensor of a single jet of shape ``[num_particles, num_features]``
          or multiple jets in Cartesian coordinates of shape ``[num_jets, num_particles, num_features]``.
        eps (float): epsilon used in the calculation. Default to 1e-12.
        return_4vec (bool): Whether to return a for vector with zero-th component being :math:`|\mathbf{p}|`. Default to False.
            If True, and ``num_features`` is 3, :math:`|\mathbf{p}|` will be calculated by :math:`| \mathbf{p} | = \sqrt{p_x^2 + p_y^2 + p_z^2}`.

    Returns:
        type(jet): 2D (if inputted single jet) or 3D array/tensor of shape in polar coordinates :math:`(\eta, \phi, p_\mathrm{T})` (if ``return_4vec`` is False) \
            or :math:`(|\mathbf{p}|, \eta, \phi, p_\mathrm{T})` (if ``return_4vec`` is True).

    Raises:
        ValueError: If the last dimension is not 3 or 4.
        NotImplementedError: The ``jet`` is not a numpy.ndarray or torch.Tensor.

    """
    if jet.shape[-1] == 4:  # (E, px, py, pz)
        idx_px, idx_py, idx_pz = 1, 2, 3
    elif jet.shape[-1] == 3:  # (px, py, pz)
        idx_px, idx_py, idx_pz = 0, 1, 2
    else:
        raise ValueError(f'Wrong last dimension of jet. Should be 3 or 4 but found: {jet.shape[-1]}.')

    px = jet[..., idx_px]
    py = jet[..., idx_py]
    pz = jet[..., idx_pz]

    if isinstance(jet, np.ndarray):
        pt = np.sqrt(px ** 2 + py ** 2 + eps)
        eta = np.arcsinh(pz / (pt + eps))
        phi = np.arctan2(py + eps, px + eps)

        if return_4vec:
            if jet.shape[-1] == 4:
                E = jet[..., 0]
            else:
                E = np.sqrt(np.sum(np.power(jet, 2), axis=-1))
            return np.stack((E, eta, phi, pt), axis=-1)
        return np.stack((eta, phi, pt), axis=-1)

    if isinstance(jet, torch.Tensor):
        pt = torch.sqrt(px ** 2 + py ** 2 + eps)
        try:
            eta = torch.asinh(pz / (pt + eps))
        except AttributeError:  # older version of torch
            eta = __arcsinh(jet)
        phi = torch.atan2(py + eps, px + eps)
        if return_4vec:
            if jet.shape[-1] == 4:
                E = jet[..., 0]
            else:
                E = torch.sqrt(torch.sum(torch.pow(jet, 2), dim=-1) + eps)
            return torch.stack((E, eta, phi, pt), dim=-1)
        return torch.stack((eta, phi, pt), dim=-1)

    raise NotImplementedError(f'Current type {jet.type} not supported. Supported types are numpy.ndarray and torch.Tensor.')


def __arcsinh(z: torch.Tensor) -> torch.Tensor:
    """Self-defined arcsinh function."""
    return torch.log(z + torch.sqrt(1 + torch.pow(z, 2)))


def get_cartesian(
    jet: np.ndarray or torch.Tensor,
    return_4vec: bool = False
) -> np.ndarray or torch.Tensor:
    """
    Convert 3- or 4-vector features in polar coordiantes to cartesian coordinates:
    :math:`(\eta, \phi, p_\mathrm{T})` or :math:`(|\mathbf{p}|, \eta, \phi, \mathrm{p}_T)`
    :math:`\longrightarrow` :math:`(p_x, p_y, p_z)` or :math:`(|\mathbf{p}|, p_x, p_y, p_z)`


    Args:
        jet (np.ndarray or torch.Tensor): array or tensor of a single jet of shape ``[num_particles, num_features]``
          or multiple jets in polar coordinates of shape ``[num_jets, num_particles, num_features]``.
        return_4vec (bool): Whether to return a for vector with zero-th component being :math:`\mathbf{p}`. Default to False.
            If True, and ``num_features`` is 3, :math:`|\mathbf{p}|` will be calculated by :math:`| \mathbf{p} | = p_\mathrm{T} \cosh\eta`.

    Returns:
        type(jet): 2D (if inputted single jet) or 3D in cartesian coordinates :math:`(p_x, p_y, p_z)` (if ``return_4vec`` is False) \
        or :math:`(|\mathbf{p}|, p_x, p_y, p_z)` (if ``return_4vec`` is False).

    Raises:
        ValueError: If the last dimension is not 3 or 4.
        NotImplementedError: The ``jet`` is not a numpy.ndarray or torch.Tensor.

    """
    if jet.shape[-1] == 4:
        idx_eta, idx_phi, idx_pt = 1, 2, 3
    elif jet.shape[-1] == 3:
        idx_eta, idx_phi, idx_pt = 0, 1, 2
    else:
        raise ValueError(f'Wrong last dimension of jet. Should be 3 or 4 but found: {jet.shape[-1]}.')

    pt = jet[..., idx_pt]
    eta = jet[..., idx_eta]
    phi = jet[..., idx_phi]

    if isinstance(jet, np.ndarray):
        px = pt * np.cos(phi)
        py = pt * np.cos(phi)
        pz = pt * np.sinh(eta)

        if return_4vec:
            if jet.shape[-1] == 4:
                E = jet[..., 0]
            else:
                E = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
            return np.stack((E, px, py, pz), axis=-1)
        return np.stack((px, py, pz), axis=-1)

    if isinstance(jet, torch.Tensor):
        px = pt * torch.cos(phi)
        py = pt * torch.cos(phi)
        pz = pt * torch.sinh(eta)
        if return_4vec:
            if jet.shape[-1] == 4:
                E = jet[..., 0]
            else:
                E = torch.sqrt(px ** 2 + py ** 2 + pz ** 2)
            return torch.stack((E, px, py, pz), dim=-1)
        return torch.stack((px, py, pz), dim=-1)

    raise NotImplementedError(f'Current type {jet.type} not supported. Supported types are numpy.ndarray and torch.Tensor.')


def get_polar_rel(
    jet: np.ndarray or torch.Tensor,
    input_cartesian: bool = False
) -> np.ndarray or torch.Tensor:
    """
    Convert 3- or 4-vector features to relative polar coordinates.
    Given jet feautures :math:`J = (J_\eta, J_\phi, J_{p_\mathrm{T}})` and particle features :math:`p = (\eta, \phi, p_\mathrm{T})`,
    the relative coordinates are given by
        .. math::
            \eta_\mathrm{T}^\mathrm{rel} = \eta - J_\eta,\
            \phi_\mathrm{T}^\mathrm{rel} = \phi - J_\phi,\
            p_\mathrm{T}^\mathrm{rel} = p_\mathrm{T} / J_{p_\mathrm{T}}

    Args:
        jet (np.ndarray or torch.Tensor): array or tensor of a single jet of shape ``[num_particles, num_features]``
          or multiple jets in cartesian or polar coordinates of shape ``[num_jets, num_particles, num_features]``.
        input_cartesian (bool): Whether ``jet`` is in Cartesian coordinate. False if ``jet`` is in polar coordinates.
          Default to False.

    Returns:
        type(jet): 2D (if inputted single jet) or 3D arrays/tensors of features in relative polar coordinates.

    Raises:
        NotImplementedError: If jet is not a numpy.ndarray or torch.Tensor.

    """
    if input_cartesian:
        jet = get_polar(jet)  # Convert to polar first
        if isinstance(jet, torch.Tensor):
            jet_features = jet.sum(dim=-2)
        elif isinstance(jet, np.ndarray):
            jet_features = jet.sum(axis=-2)
        else:
            raise NotImplementedError(f'Current type {jet.type} not supported. Supported types are numpy.ndarray and torch.Tensor.')
    else:
        if isinstance(jet, torch.Tensor):
            jet_features = get_polar(get_cartesian(jet).sum(dim=-2))
        elif isinstance(jet, np.ndarray):
            jet_features = get_polar(get_cartesian(jet).sum(axis=-2))
        else:
            raise NotImplementedError(f'Current type {jet.type} not supported. Supported types are numpy.ndarray and torch.Tensor.')

    idx_eta, idx_phi, idx_pt = 0, 1, 2
    pt = jet[..., idx_pt]
    eta = jet[..., idx_eta]
    phi = jet[..., idx_phi]

    num_particles = jet.shape[-2]
    if isinstance(jet, np.ndarray):
        pt /= np.repeat(np.expand_dims(jet_features[..., idx_pt], axis=-1), num_particles, axis=-1)
        eta -= np.repeat(np.expand_dims(jet_features[..., idx_eta], axis=-1), num_particles, axis=-1)
        phi -= np.repeat(np.expand_dims(jet_features[..., idx_phi], axis=-1), num_particles, axis=-1)
        return np.stack((eta, phi, pt), axis=-1)

    pt /= jet_features[..., idx_pt].unsqueeze(dim=-1).repeat(1, num_particles)
    eta -= jet_features[..., idx_eta].unsqueeze(dim=-1).repeat(1, num_particles)
    phi -= jet_features[..., idx_phi].unsqueeze(dim=-1).repeat(1, num_particles)
    return torch.stack((eta, phi, pt), dim=-1)
