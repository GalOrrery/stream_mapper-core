"""Core feature."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from stream_ml.core.data import Data
from stream_ml.core.utils.scale._api import DataScaler, T

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.typing import Array


###############################################################################


class StandardScaler(DataScaler):
    """Standardize features by removing the mean and scaling to unit variance."""

    mean: Any
    scale: Any

    def fit(self, data: Any, names: tuple[str, ...]) -> StandardScaler:
        """Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        data : Array, positional-only
            The data used to compute the mean and standard deviation.
        names : tuple[str, ...], optional
            The names of the columns to scale.

        Returns
        -------
        StandardScaler
        """
        if isinstance(data, Data):
            data = data[names].array

        ncols = data.shape[1]
        self.mean = np.array([np.nanmean(np.array(data[:, i])) for i in range(ncols)])
        self.scale = np.array([np.nanstd(np.array(data[:, i])) for i in range(ncols)])
        self.names = names

        return self

    def transform(self, data: T, /, names: tuple[str, ...]) -> T:
        """Standardize a dataset along the features axis."""
        mean = np.array([self.mean[self.names.index(name)] for name in names])
        scale = np.array([self.scale[self.names.index(name)] for name in names])

        return cast(T, _transform(data, mean, scale, names=names))

    def fit_transform(self, data: T, /, names: tuple[str, ...]) -> T:
        """Fit to data, then transform it."""
        return self.fit(data, names=names).transform(data, names=names)

    def inverse_transform(
        self,
        data: T,
        /,
        names: tuple[str, ...],
        **kwargs: Any,
    ) -> T:
        """Scale back the data to the original representation."""
        return cast(
            T,
            _transform_inv(data, self.mean, self.scale, names=names),
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
    data: Data[Array],
    mean: Array,
    scale: Array,
    /,
    names: tuple[str, ...],
) -> Data[Array]:
    shape = (-1,) + (1,) * (len(data.array.shape) - 2)
    mean_ = np.array([mean[data.names.index(name)] for name in names]).reshape(shape)
    scale_ = np.array([scale[data.names.index(name)] for name in names]).reshape(shape)
    return Data(
        _transform_inv(data[names].array, mean_, scale_, names=names),
        names=names,
    )
