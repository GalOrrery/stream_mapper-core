"""Core feature."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, cast

import numpy as np

__all__: list[str] = []

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from stream_ml.core.typing.array import Array


@singledispatch
def within_bounds(
    value: Array,
    /,
    lower_bound: Array | float | None,
    upper_bound: Array | float | None,
    *,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> Array:
    """Check if a value is within the given bounds.

    Parameters
    ----------
    value : ndarray
        Value to check.
    lower_bound, upper_bound : float | None
        Bounds to check against.
    lower_inclusive, upper_inclusive : bool, optional
        Whether to include the bounds in the check, by default `True`.

    Returns
    -------
    ndarray
        Boolean array indicating whether the value is within the bounds.
    """
    raise NotImplementedError


####################################################################################################


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def fit(self, X: NDArray[Any], /) -> StandardScaler:  # noqa: N803
        """Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : Array, positional-only
            The data used to compute the mean and standard deviation.

        Returns
        -------
        StandardScaler
        """
        ncols = X.shape[1]

        self.mean_ = np.array([np.nanmean(X[:, i]) for i in range(ncols)])
        self.scale_ = np.array([np.nanstd(X[:, i]) for i in range(ncols)])

        return self

    def transform(self, X: NDArray[Any], /) -> NDArray[Any]:  # noqa: N803
        """Standardize a dataset along the features axis."""
        return cast("NDArray[Any]", (X - self.mean_) / self.scale_)

    def fit_transform(self, X: NDArray[Any], /) -> NDArray[Any]:  # noqa: N803
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)
