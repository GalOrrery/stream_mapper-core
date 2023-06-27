"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass, replace
from functools import singledispatch
from typing import Any, Generic, cast

import numpy as np

from stream_ml.core.data import Data
from stream_ml.core.typing import Array
from stream_ml.core.utils.compat import get_namespace
from stream_ml.core.utils.scale._api import DataScaler, T

###############################################################################


@dataclass(frozen=True)
class StandardScaler(DataScaler, Generic[Array]):
    """Standardize features by removing the mean and scaling to unit variance."""

    mean: Array
    scale: Array
    names: tuple[str, ...]

    @classmethod
    def fit(cls, data: Any, names: tuple[str, ...]) -> StandardScaler[Array]:
        """Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        data : Array, positional-only
            The data used to compute the mean and standard deviation.
        names : tuple[str, ...], optional
            The names of columns in `data`.

        Returns
        -------
        StandardScaler
        """
        if isinstance(data, Data):
            data = data[names].array

        xp = get_namespace(data)
        return cls(mean=xp.mean(data, 0), scale=xp.std(data, 0), names=names)

    def transform(self, data: T, /, names: tuple[str, ...]) -> T:
        """Standardize a dataset along the features axis."""
        mean = self.mean[[self.names.index(n) for n in names]]
        scale = self.scale[[self.names.index(n) for n in names]]

        return cast(T, _transform(data, mean, scale, names=names))

    def inverse_transform(
        self,
        data: T,
        /,
        names: tuple[str, ...],
        **kwargs: Any,
    ) -> T:
        """Scale back the data to the original representation."""
        return cast(T, _transform_inv(data, self.mean, self.scale, names=names))

    def __getitem__(self, names: tuple[str, ...]) -> StandardScaler[Array]:
        """Get a subset DataScaler with the given names."""
        return replace(
            self,
            mean=self.mean[[self.names.index(n) for n in names]],
            scale=self.scale[[self.names.index(n) for n in names]],
            names=names,
        )


# ============================================================================
# Transform


@singledispatch
def _transform(data: Any, mean: Any, scale: Any, /, names: tuple[str, ...]) -> Any:
    """Standardize a dataset along each features axis."""
    return (data - mean) / scale


@_transform.register(Data)
def _transform_data(
    data: Data[Array], mean: Array, scale: Array, /, names: tuple[str, ...]
) -> Data[Array]:
    array = data[names].array
    # Need to adjust the mean to be the same shape
    mean = mean[(None, slice(None)) + (None,) * (len(array.shape) - 2)]
    scale = scale[(None, slice(None)) + (None,) * (len(array.shape) - 2)]
    return Data(_transform(array, mean, scale, names=names), names=names)


# ============================================================================
# Inverse transform


@singledispatch
def _transform_inv(data: Any, mean: Any, scale: Any, /, names: tuple[str, ...]) -> Any:
    """Scale back the data to the original representation."""
    return data * scale + mean


@_transform_inv.register(Data)
def _transform_inv_data(
    data: Data[Array], mean: Array, scale: Array, /, names: tuple[str, ...]
) -> Data[Array]:
    shape = (-1,) + (1,) * (len(data.array.shape) - 2)
    mean_ = np.array([mean[data.names.index(name)] for name in names]).reshape(shape)
    scale_ = np.array([scale[data.names.index(name)] for name in names]).reshape(shape)
    return Data(
        _transform_inv(data[names].array, mean_, scale_, names=names),
        names=names,
    )
