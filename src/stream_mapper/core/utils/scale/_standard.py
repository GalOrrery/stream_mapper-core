"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass, replace
from functools import singledispatch
from typing import Any, cast, overload

import numpy as np

from stream_mapper.core._data import Data
from stream_mapper.core.typing import Array, ArrayNamespace
from stream_mapper.core.utils.compat import get_namespace
from stream_mapper.core.utils.scale._api import DataScaler

###############################################################################


@dataclass(frozen=True)
class StandardScaler(DataScaler[Array]):
    """Standardize features by removing the mean and scaling to unit variance."""

    mean: Array
    scale: Array
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple):
            object.__setattr__(self, "names", tuple(self.names))  # type: ignore[unreachable]

    @classmethod
    def fit(cls, data: Any, names: tuple[str, ...]) -> StandardScaler[Array]:
        """Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        data : (N, F) Array, positional-only
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
        return cls(
            mean=xp.asarray(np.nanmean(data, 0)),
            scale=xp.asarray(np.nanstd(data, 0)),
            names=names,
        )

    # ---------------------------------------------------------------

    @overload
    def transform(
        self,
        data: Data[Array] | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array]: ...

    @overload
    def transform(
        self,
        data: Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Array: ...

    def transform(
        self,
        data: Data[Array] | Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array] | Array:
        """Standardize a dataset along the features axis."""
        mean = self.mean[[self.names.index(n) for n in names]]
        scale = self.scale[[self.names.index(n) for n in names]]
        return cast(
            Data[Array] | Array, _transform(data, mean, scale, names=names, xp=xp)
        )

    # ---------------------------------------------------------------

    @overload
    def inverse_transform(
        self,
        data: Data[Array],
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array]: ...

    @overload
    def inverse_transform(
        self,
        data: Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Array: ...

    def inverse_transform(
        self,
        data: Data[Array] | Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array] | Array:
        """Scale back the data to the original representation."""
        return cast(
            Data[Array] | Array,
            _transform_inv(data, self.mean, self.scale, names=names, xp=xp),
        )

    # ---------------------------------------------------------------

    def __getitem__(self, names: str | tuple[str, ...]) -> StandardScaler[Array]:
        """Get a subset DataScaler with the given names."""
        names_tuple = (names,) if isinstance(names, str) else names
        return replace(
            self,
            mean=self.mean[[self.names.index(n) for n in names_tuple]],
            scale=self.scale[[self.names.index(n) for n in names_tuple]],
            names=names_tuple,
        )


# ============================================================================
# Transform


@singledispatch
def _transform(
    data: Any,
    mean: Any,
    scale: Any,
    /,
    names: tuple[str, ...],
    *,
    xp: ArrayNamespace[Array] | None,
) -> Any:
    """Standardize a dataset along each features axis."""
    return (data - mean) / scale


@_transform.register(Data)
def _transform_data(
    data: Data[Array],
    mean: Array,
    scale: Array,
    /,
    names: tuple[str, ...],
    *,
    xp: ArrayNamespace[Array] | None,
) -> Data[Array]:
    array = data[names].array
    # Need to adjust the mean to be the same shape
    mean = mean[(None, slice(None)) + (None,) * (len(array.shape) - 2)]
    scale = scale[(None, slice(None)) + (None,) * (len(array.shape) - 2)]
    return Data(_transform(array, mean, scale, names=names, xp=xp), names=names)


# ============================================================================
# Inverse transform


@singledispatch
def _transform_inv(
    data: Any,
    mean: Any,
    scale: Any,
    /,
    names: tuple[str, ...],
    *,
    xp: ArrayNamespace[Array] | None,
) -> Any:
    """Scale back the data to the original representation."""
    return data * scale + mean


@_transform_inv.register(Data)
def _transform_inv_data(
    data: Data[Array],
    mean: Array,
    scale: Array,
    /,
    names: tuple[str, ...],
    *,
    xp: ArrayNamespace[Array] | None,
) -> Data[Array]:
    shape = (-1,) + (1,) * (len(data.array.shape) - 2)
    mean_ = np.array([mean[data.names.index(name)] for name in names]).reshape(shape)
    scale_ = np.array([scale[data.names.index(name)] for name in names]).reshape(shape)
    return Data(
        _transform_inv(data[names].array, mean_, scale_, names=names, xp=xp),
        names=names,
    )
