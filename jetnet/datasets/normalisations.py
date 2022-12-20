"""
Suite of common ways to normalise data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike


class NormaliseABC(ABC):
    """
    ABC for generalised normalisation class.
    """

    def features_need_deriving(self) -> bool:
        """Checks if any dataset values or features need to be derived"""
        return False

    def derive_dataset_features(self, x: ArrayLike):
        """Derive features from dataset needed for normalisation if needed"""
        pass

    @abstractmethod
    def __call__(self, x: ArrayLike, inverse: bool = False, inplace: bool = False) -> ArrayLike:
        """
        Normalises (``inverse`` = False) or inverses normalisation of (``inverse`` = True) ``x``
        Performed inplace if ``inplace`` is True.
        """
        pass


class FeaturewiseLinear(NormaliseABC):
    """
    Shifts features by ``feature_shifts`` then multiplies by ``feature_scales``.

    If using the ``normal`` option, ``feature_shifts`` and ``feature_scales`` can be derived from
    the dataset (by calling ``derive_dataset_features``) to normalise the data to have 0 mean and
    unit standard deviation per feature.

    Args:
        feature_shifts (Union[float, List[float]], optional): value to shift features by.
            Can either be a single float for all features, or a list of length ``num_features``.
            Defaults to 0.0.
        feature_scales (Union[float, List[float]], optional): after shifting, value to multiply
            features by. Can either be a single float for all features, or a list of length
            ``num_features``. Defaults to 1.0.
        normalise_features (Optional[List[bool]], optional): if only some features need to be
            normalised, can input here a list of booleans of length ``num_features`` with ``True``
            meaning normalise and ``False`` meaning to ignore. Defaults to None i.e. normalise all.
        normal (bool, optional): derive ``feature_shifts`` and ``feature_scales`` to have 0 mean and
            unit standard deviation per feature after normalisation (``derive_dataset_features``
            method must be called before normalising).

    """

    def __init__(
        self,
        feature_shifts: Union[float, List[float]] = 0.0,
        feature_scales: Union[float, List[float]] = 1.0,
        normalise_features: Optional[List[bool]] = None,
        normal: bool = False,
    ):
        super().__init__()

        self.feature_shifts = feature_shifts
        self.feature_scales = feature_scales
        self.normalise_features = normalise_features
        self.normal = normal

    def derive_dataset_features(self, x: ArrayLike) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        If using the ``normal`` option, this will derive the means and standard deviations per
        feature, and save and return them. If not, will do nothing.

        Args:
            x (ArrayLike): dataset of shape [..., ``num_features``].

        Returns:
            (Optional[Tuple[np.ndarray, np.ndarray]]): if ``normal`` option, means and stds of each
                feature.

        """
        if self.normal:
            num_features = x.shape[-1]
            self.feature_shifts = -np.mean(x.reshape(-1, num_features), axis=0)
            self.feature_scales = 1.0 / np.std(x.reshape(-1, num_features), axis=0)
            return self.feature_shifts, self.feature_scales

    def features_need_deriving(self) -> bool:
        """Checks if any dataset values or features need to be derived"""
        return (self.feature_shifts is None) or (self.feature_scales is None)

    def __call__(self, x: ArrayLike, inverse: bool = False, inplace: bool = False) -> ArrayLike:
        assert not self.features_need_deriving(), (
            "Feature means and stds have not been specified, "
            + "you need to either set or derive them first"
        )

        num_features = x.shape[-1]

        if isinstance(self.feature_shifts, float):
            feature_shifts = np.full(num_features, self.feature_shifts)
        else:
            feature_shifts = self.feature_shifts

        if isinstance(self.feature_scales, float):
            feature_scales = np.full(num_features, self.feature_scales)
        else:
            feature_scales = self.feature_scales

        if self.normalise_features is None:
            normalise_features = np.full(num_features, True)
        elif isinstance(self.normalise_features, bool):
            normalise_features = np.full(num_features, self.normalise_features)
        else:
            normalise_features = self.normalise_features

        assert (
            len(feature_shifts) == num_features
        ), "Number of features in input does not equal number of specified feature shifts"

        assert (
            len(feature_scales) == num_features
        ), "Number of features in input does not equal number of specified feature scales"

        assert (
            len(normalise_features) == num_features
        ), "Number of features in input does not equal length of ``normalise_features``"

        if not inplace:
            if isinstance(x, torch.Tensor):
                x = torch.clone(x)
            else:
                x = np.copy(x)

        if not inverse:
            for i in range(num_features):
                if normalise_features[i]:
                    x[..., i] += feature_shifts[i]
                    x[..., i] *= feature_scales[i]

        else:
            for i in range(num_features):
                if normalise_features[i]:
                    x[..., i] /= feature_scales[i]
                    x[..., i] -= feature_shifts[i]

        return x

    def __repr__(self) -> str:
        if self.normal:
            ret = "Normalising features to zero mean and unit standard deviation"
        else:
            ret = (
                f"Shift features by {self.feature_shifts} "
                f"and then multiplying by {self.feature_scales}"
            )

        if self.normalise_features is not None and self.normalise_features is not True:
            ret += f", normalising features: {self.normalise_features}"

        return ret


class FeaturewiseLinearBounded(NormaliseABC):
    """
    Normalizes dataset features by scaling each to an (absolute) max of ``feature_norms``
    and shifting by ``feature_shifts``.

    If the value in the list for a feature is None, it won't be scaled or shifted.

    Args:
        feature_norms (Union[float, List[float]], optional): max value to scale each feature to.
            Can either be a single float for all features, or a list of length ``num_features``.
            Defaults to 1.0.
        feature_shifts (Union[float, List[float]], optional): after scaling,
            value to shift feature by.
            Can either be a single float for all features, or a list of length ``num_features``.
            Defaults to 0.0.
        feature_maxes (List[float], optional): max pre-scaling absolute value of each feature, used
            for scaling to the norm and inverting.
        normalise_features (Optional[List[bool]], optional): if only some features need to be
            normalised, can input here a list of booleans of length ``num_features`` with ``True``
            meaning normalise and ``False`` meaning to ignore. Defaults to None i.e. normalise all.

    """

    def __init__(
        self,
        feature_norms: Union[float, List[float]] = 1.0,
        feature_shifts: Union[float, List[float]] = 0.0,
        feature_maxes: List[float] = None,
        normalise_features: Optional[List[bool]] = None,
    ):
        super().__init__()

        self.feature_norms = feature_norms
        self.feature_shifts = feature_shifts
        self.feature_maxes = feature_maxes
        self.normalise_features = normalise_features

    def derive_dataset_features(self, x: ArrayLike) -> np.ndarray:
        """
        Derives, saves, and returns absolute feature maxes of dataset ``x``.

        Args:
            x (ArrayLike): dataset of shape [..., ``num_features``].

        Returns:
            np.ndarray: feature maxes

        """
        num_features = x.shape[-1]
        self.feature_maxes = np.max(np.abs(x.reshape(-1, num_features)), axis=0)
        return self.feature_maxes

    def features_need_deriving(self) -> bool:
        """Checks if any dataset values or features need to be derived"""
        return self.feature_maxes is None

    def __call__(self, x: ArrayLike, inverse: bool = False, inplace: bool = False) -> ArrayLike:
        assert (
            not self.features_need_deriving()
        ), "Feature maxes have not been specified, you need to either set or derive them first"

        num_features = x.shape[-1]

        assert num_features == len(
            self.feature_maxes
        ), "Number of features in ``x`` does not equal length of saved feature maxes"

        if isinstance(self.feature_norms, float):
            feature_norms = np.full(num_features, self.feature_norms)
        else:
            feature_norms = self.feature_norms

        if isinstance(self.feature_shifts, float):
            feature_shifts = np.full(num_features, self.feature_shifts)
        else:
            feature_shifts = self.feature_shifts

        if self.normalise_features is None:
            normalise_features = np.full(num_features, True)
        elif isinstance(self.normalise_features, bool):
            normalise_features = np.full(num_features, self.normalise_features)
        else:
            normalise_features = self.normalise_features

        assert (
            len(feature_shifts) == num_features
        ), "Number of features in input does not equal number of specified feature shifts"

        assert (
            len(feature_norms) == num_features
        ), "Number of features in input does not equal number of specified feature norms"

        assert (
            len(normalise_features) == num_features
        ), "Number of features in input does not equal length of ``normalise_features``"

        if not inplace:
            if isinstance(x, torch.Tensor):
                x = torch.clone(x)
            else:
                x = np.copy(x)

        if not inverse:
            for i in range(num_features):
                if normalise_features[i]:
                    if feature_norms[i] is not None:
                        x[..., i] /= self.feature_maxes[i]
                        x[..., i] *= feature_norms[i]

                    if feature_shifts[i] is not None:
                        x[..., i] += feature_shifts[i]

        else:
            for i in range(num_features):
                if normalise_features[i]:
                    if feature_shifts[i] is not None:
                        x[..., i] -= feature_shifts[i]

                    if feature_norms[i] is not None:
                        x[..., i] /= feature_norms[i]
                        x[..., i] *= self.feature_maxes[i]

        return x

    def __repr__(self) -> str:
        ret = (
            f"Linear scaling features to feature norms {self.feature_norms} "
            f" and (post-scaling) feature shifts {self.feature_shifts}"
        )

        if self.feature_maxes is not None:
            ret += f", with pre-scaling feature maxes {self.feature_maxes}"

        if self.normalise_features is not None and self.normalise_features is not True:
            ret += f", normalising features: {self.normalise_features}"

        return ret
