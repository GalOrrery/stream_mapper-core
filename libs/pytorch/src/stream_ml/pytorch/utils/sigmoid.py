"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
from torch import nn

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array

__all__: list[str] = []

_0 = xp.asarray(0)
_1 = xp.asarray(1)


def scaled_sigmoid(x: Array, /, lower: Array = _0, upper: Array = _1) -> Array:
    """Sigmoid function mapping ``(-inf, inf)`` to ``(lower, upper)``.

    Output for (lower, upper) is defined as:
    - If (finite, finite), then this is a scaled sigmoid function.
    - If (-inf, inf) then this is the identity function.
    - Not implemented for (+/- inf, any), (any, +/- inf)

    Parameters
    ----------
    x : Array
        X.
    lower : Array
        Lower.
    upper : Array
        Upper.

    Returns
    -------
    Array

    See Also
    --------
    stream_ml.core.utils.map_to_range
        Maps ``[min(x), max(x)]`` to range ``[lower, upper]``.
    """
    if xp.isneginf(lower) and xp.isposinf(upper):
        return x
    elif xp.isinf(lower) or xp.isinf(upper):
        raise NotImplementedError

    return xp.sigmoid(x) * (upper - lower) + lower


class ColumnarScaledSigmoid(nn.Module):  # type: ignore[misc]
    r"""Applies scaled sigmoid function to the fraction and sigma."""

    __constants__ = ["columns", "bounds", "inplace"]
    columns: tuple[int, ...]
    bounds: tuple[tuple[Array, Array], ...]
    inplace: bool

    def __init__(
        self,
        columns: tuple[int, ...],
        bounds: tuple[tuple[float | Array, float | Array], ...],
        *,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.bounds = tuple((xp.asarray(a), xp.asarray(b)) for a, b in bounds)
        self.inplace = inplace

        if len(columns) != len(bounds):
            msg = "columns and bounds must be the same length"
            raise ValueError(msg)

    def forward(self, arr: Array) -> Array:
        """Forward pass."""
        if not self.inplace:
            arr = arr.clone()

        for col, (lower, upper) in zip(self.columns, self.bounds, strict=True):
            arr[:, col] = scaled_sigmoid(arr[:, col], lower=lower, upper=upper)

        return arr
