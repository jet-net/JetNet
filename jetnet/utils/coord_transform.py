import numpy as np
import torch
from typing import Union, Iterable


def p4_polar_from_p4_cartesian_massless(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from Cartesian coordinates to polar coordinates for massless particle features.

    Args:
        p4 (np.ndarray or torch.Tensor): array of 4-momenta in Cartesian coordinates,
          of shape ``[..., 4]``. The last axis should be in order
          :math:`(E/c, p_x, p_y, p_z)`.

    Returns:
        np.ndarray or torch.Tensor: array of 4-momenta in polar coordinates, arranged in order
          :math:`(E/c, p_\mathrm{T}, \eta, \phi)`, where :math:`\eta` is the pseudorapidity.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # (E/c, px, py, pz) -> (E/c, pT, eta, phi)
    p0, px, py, pz = __unbind(p4, axis=-1)
    pt = __sqrt(px**2 + py**2)
    eta = __arcsinh(pz / (pt + eps))
    phi = __arctan2(py, px)

    return __stack([p0, pt, eta, phi], axis=-1)


def p4_cartesian_from_p4_polar_massless(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from polar coordinates to Cartesian coordinates for massless particle features.

    Args:
        p4 (np.ndarray or torch.Tensor): array of 4-momenta in polar coordinates,
          of shape ``[..., 4]``. The last axis should be in order
          :math:`(E/c, p_\mathrm{T}, \eta, \phi)`, where :math:`\eta` is the pseudorapidity.

    Returns:
        np.ndarray or torch.Tensor: array of 4-momenta in polar coordinates, arranged in order
          :math:`(E/c, p_x, p_y, p_z)`.
    """

    # (E/c, pT, eta, phi) -> (E/c, px, py, pz)
    p0, pt, eta, phi = __unbind(p4, axis=-1)
    px = pt * __cos(phi)
    py = pt * __sin(phi)
    pz = pt * __sinh(eta)

    return __stack([p0, px, py, pz], axis=-1)


def p4_polar_from_p4_cartesian(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from Cartesian coordinates to polar coordinates.

    Args:
        p4 (np.ndarray or torch.Tensor): array of 4-momenta in Cartesian coordinates,
          of shape ``[..., 4]``. The last axis should be in order
          :math:`(E/c, p_x, p_y, p_z)`.

    Returns:
        np.ndarray or torch.Tensor: array of 4-momenta in polar coordinates, arranged in order
          :math:`(E/c, p_\mathrm{T}, y, \phi)`, where :math:`y` is the rapidity.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # (E/c, p_x, p_y, p_z) -> (E/c, p_T, y, phi)
    p0, px, py, pz = __unbind(p4, axis=-1)
    pt = __sqrt(px**2 + py**2)
    y = 0.5 * __log((p0 + pz + eps) / (p0 - pz + eps))
    phi = __arctan2(py, px)

    return __stack([p0, pt, y, phi], axis=-1)


def p4_cartesian_from_p4_polar(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform 4-momenta from polar coordinates to Cartesian coordinates.

    Args:
        p4 (np.ndarray or torch.Tensor): array of 4-momenta in Cartesian coordinates,
          of shape ``[..., 4]``. The last axis should be in order
          :math:`(E/c, p_\mathrm{T}, y, \phi)`, where :math:`y` is the rapidity.

    Returns:
        np.ndarray or torch.Tensor: array of 4-momenta in polar coordinates, arranged in order
          :math:`(E/c, p_\mathrm{T}, y, \phi)`, where :math:`y` is the rapidity.
          :math:`(E/c, p_x, p_y, p_z)`.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # (E/c, pT, y, phi) -> (E/c, px, py, pz)
    p0, pt, y, phi = __unbind(p4, axis=-1)
    px = pt * __cos(phi)
    py = pt * __sin(phi)
    # get pz
    mt = p0 / (__cosh(y) + eps)  # get transverse mass
    pz = mt * __sinh(y)
    return __stack([p0, px, py, pz], axis=-1)


def polarrel_from_p4_cartesian(
    p4: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get particle features in relative polar coordinates from 4-momenta in Cartesian coordinates.

    Args:
        p4 (np.ndarray or torch.Tensor): array of 4-momenta in Cartesian coordinates,
          of shape ``[..., 4]``. The last axis should be in order
          :math:`(E/c, p_x, p_y, p_z)`.

    Returns:
        np.ndarray or torch.Tensor: array of features in relative polar coordinates, arranged in order
          :math:`(p_\mathrm{T}^\mathrm{rel}, \eta^\mathrm{rel}, \phi^\mathrm{rel})`.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # particle (pT, eta, phi)
    p4_polar = p4_polar_from_p4_cartesian_massless(p4)
    _, pt, eta, phi = __unbind(p4_polar, axis=-1)

    # jet (PT, Eta, Phi)
    jet_cartesian = __sum(p4, axis=-2, keepdims=True)
    jet_polar = p4_polar_from_p4_cartesian(jet_cartesian)
    _, Pt, Eta, Phi = __unbind(jet_polar, axis=-1)

    # get relative features
    pt_rel = pt / (Pt + eps)
    eta_rel = eta - Eta
    phi_rel = phi - Phi
    phi_rel = (phi_rel + np.pi) % (2 * np.pi) - np.pi  # map to [-pi, pi]

    return __stack([pt_rel, eta_rel, phi_rel], axis=-1)


def polarrel_from_p4_polar(p4: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Get particle features in relative polar coordinates from 4-momenta in polar coordinates.

    Args:
        p4 (np.ndarray or torch.Tensor): array of 4-momenta in polar coordinates,
          of shape ``[..., 4]``. The last axis should be in order
          :math:`(E/c, p_\mathrm{T}, \eta, \phi)`, where :math:`\eta` is the pseudorapidity.

    Returns:
        np.ndarray or torch.Tensor: array of features in relative polar coordinates, arranged in order
        :math:`(p_\mathrm{T}^\mathrm{rel}, \eta^\mathrm{rel}, \phi^\mathrm{rel})`.
    """

    eps = __get_default_eps(p4)  # default epsilon for the dtype

    # particle (pT, eta, phi)
    p4_polar = p4
    _, pt, eta, phi = __unbind(p4_polar, axis=-1)

    # jet (PT, Eta, Phi)
    p4 = p4_cartesian_from_p4_polar(p4_polar)
    # expand dimension to (..., 1, 4) to match p4 shape
    jet_cartesian = __sum(p4, axis=-2, keepdims=True)
    jet_polar = p4_polar_from_p4_cartesian(jet_cartesian)
    _, Pt, Eta, Phi = __unbind(jet_polar, axis=-1)

    # get relative features
    pt_rel = pt / (Pt + eps)
    eta_rel = eta - Eta
    phi_rel = phi - Phi
    phi_rel = (phi_rel + np.pi) % (2 * np.pi) - np.pi  # map to [-pi, pi]

    return __stack([pt_rel, eta_rel, phi_rel], axis=-1)


def p4_polar_from_polarrel(
    p_polarrel: Union[np.ndarray, torch.Tensor],
    jet_features: Union[np.ndarray, torch.Tensor],
    jet_coord: str = "cartesian",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get particle features in absolute polar coordinates from relative polar coordinates and jet features.

    Args:
        p_polarrel (np.ndarray or torch.Tensor): array of particle features in relative polar coordinates,
          of shape ``[..., 3]``. The last axis should be in order
          :math:`(p_\mathrm{T}^\mathrm{rel}, \eta^\mathrm{rel}, \phi^\mathrm{rel})`, where :math:`\eta` is the pseudorapidity.
        jet_features (np.ndarray or torch.Tensor): array of jet features in polar coordinates,
          of shape ``[..., 4]``. The coordinates are specified by ``jet_coord``.
        jet_coord (str): coordinate system of jet features. Can be either "cartesian" or "polar".
            Defaults to "cartesian".
            If "cartesian", the last axis of ``jet_features`` should be in order :math:`(E/c, p_x, p_y, p_z)`.
            If "polar", the last axis of ``jet_features`` should be in order :math:`(E/c, p_\mathrm{T}, \eta, \phi)`.

    Returns:
        np.ndarray or torch.Tensor: array of particle features in absolute polar coordinates, arranged in order
          :math:`(E/c, p_\mathrm{T}, \eta, \phi)`.
    """

    # particle features
    pt_rel, eta_rel, phi_rel = __unbind(p_polarrel, axis=-1)

    # jet features
    if jet_coord.lower() == "cartesian":
        jet_features = p4_polar_from_p4_cartesian(jet_features)
    elif jet_coord.lower() == "polar":
        pass
    else:
        raise ValueError("jet_coord can only be 'cartesian' or 'polar'")
    # eta is used even though jet is massive
    _, Pt, Eta, Phi = __unbind(jet_features, axis=-1)

    # transform back to absolute coordinates
    pt = pt_rel * Pt
    eta = eta_rel + Eta
    phi = phi_rel + Phi
    p0 = pt * __cosh(eta)

    return __stack([p0, pt, eta, phi], axis=-1)


def p4_cartesian_from_polarrel(
    p_polarrel: Union[np.ndarray, torch.Tensor],
    jet_features: Union[np.ndarray, torch.Tensor],
    jet_coord: str = "cartesian",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Get particle features in absolute Cartesian coordinates from relative polar coordinates and jet features.

    Args:
        p_polarrel (np.ndarray or torch.Tensor): array of particle features in relative polar coordinates,
          of shape ``[..., 3]``. The last axis should be in order
          :math:`(p_\mathrm{T}^\mathrm{rel}, \eta^\mathrm{rel}, \phi^\mathrm{rel})`, where :math:`\eta` is the pseudorapidity.
        jet_features (np.ndarray or torch.Tensor): array of jet features in polar coordinates,
          of shape ``[..., 4]``. The coordinates are specified by ``jet_coord``.
        jet_coord (str): coordinate system of jet features. Can be either "cartesian" or "polar".
            Defaults to "cartesian".
            If "cartesian", the last axis of ``jet_features`` should be in order :math:`(E/c, p_x, p_y, p_z)`.
            If "polar", the last axis of ``jet_features`` should be in order :math:`(E/c, p_\mathrm{T}, \eta, \phi)`.

    Returns:
        np.ndarray or torch.Tensor: array of particle features in absolute polar coordinates, arranged in order
          :math:`(E/c, p_x, p_y, p_z)`.
    """
    p4_polar = p4_polar_from_polarrel(p_polarrel, jet_features, jet_coord)
    # eta is used even though jet is massive
    p4_cartesian = p4_cartesian_from_p4_polar_massless(p4_polar)
    return p4_cartesian


def __unbind(x: Union[np.ndarray, torch.Tensor], axis: int) -> Union[np.ndarray, torch.Tensor]:
    """Unbind an np.ndarray or torch.Tensor along a given axis."""
    if isinstance(x, torch.Tensor):
        return torch.unbind(x, dim=axis)
    elif isinstance(x, np.ndarray):
        return np.rollaxis(x, axis=axis)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __stack(
    x: Iterable[Union[np.ndarray, torch.Tensor]], axis: int
) -> Union[np.ndarray, torch.Tensor]:
    """Stack an iterable of np.ndarray or torch.Tensor along a given axis."""
    if not isinstance(x, Iterable):
        raise TypeError("x must be an iterable.")

    if isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=axis)
    elif isinstance(x[0], np.ndarray):
        return np.stack(x, axis=axis)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __cos(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Cosine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.cos(x)
    elif isinstance(x, np.ndarray):
        return np.cos(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __sin(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Sine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.sin(x)
    elif isinstance(x, np.ndarray):
        return np.sin(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __sinh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Hyperbolic sine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.sinh(x)
    elif isinstance(x, np.ndarray):
        return np.sinh(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __arcsinh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Inverse hyperbolic sine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.asinh(x)
    elif isinstance(x, np.ndarray):
        return np.arcsinh(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __cosh(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Hyperbolic cosine function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.cosh(x)
    elif isinstance(x, np.ndarray):
        return np.cosh(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __log(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Logarithm function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.log(x)
    elif isinstance(x, np.ndarray):
        return np.log(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __arctan2(
    y: Union[np.ndarray, torch.Tensor], x: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """Arctangent function that works with np.ndarray and torch.Tensor."""
    if isinstance(y, torch.Tensor):
        return torch.atan2(y, x)
    elif isinstance(y, np.ndarray):
        return np.arctan2(y, x)
    else:
        raise TypeError(f"y must be either a numpy array or a torch tensor, not {type(y)}")


def __sqrt(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Square root function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return torch.sqrt(x)
    elif isinstance(x, np.ndarray):
        return np.sqrt(x)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __sum(
    x: Union[np.ndarray, torch.Tensor], axis: int, keepdims: bool = False
) -> Union[np.ndarray, torch.Tensor]:
    """Sum function that works with np.ndarray and torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.sum(axis, keepdim=keepdims)
    elif isinstance(x, np.ndarray):
        return np.sum(x, axis=axis, keepdims=keepdims)
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


def __get_default_eps(x: Union[np.ndarray, torch.Tensor]) -> float:
    if isinstance(x, torch.Tensor):
        return torch.finfo(x.dtype).eps
    elif isinstance(x, np.ndarray):
        return np.finfo(x.dtype).eps
    else:
        raise TypeError(f"x must be either a numpy array or a torch tensor, not {type(x)}")


__ALL__ = [
    p4_polar_from_p4_cartesian_massless,
    p4_cartesian_from_p4_polar_massless,
    p4_polar_from_p4_cartesian,
    p4_cartesian_from_p4_polar,
    polarrel_from_p4_cartesian,
    polarrel_from_p4_polar,
]
