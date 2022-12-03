"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
import torch.nn as nn

if TYPE_CHECKING:
    # LOCAL
    from stream_ml._typing import Array

__all__: list[str] = []


def sigmoid(x: Array, /, lower: Array | float = 0, upper: Array | float = 1) -> Array:
    """Sigmoid function then scaling to within (lower, upper).

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
    """
    return xp.sigmoid(x) * (upper - lower) + lower


class ColumnarScaledSigmoid(nn.Module):  # type: ignore[misc]
    r"""Applies scaled sigmoid function to the fraction and sigma."""

    __constants__ = ["columns", "bounds", "inplace"]
    columns: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]
    inplace: bool

    def __init__(
        self,
        columns: tuple[int, ...],
        bounds: tuple[tuple[float, float], ...],
        *,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.bounds = bounds
        self.inplace = inplace

        if len(columns) != len(bounds):
            raise ValueError("columns and bounds must be the same length")

    def forward(self, arr: Array) -> Array:
        """Forward pass."""
        if not self.inplace:
            arr = arr.clone()

        for col, (lower, upper) in zip(self.columns, self.bounds):
            arr[:, col] = sigmoid(arr[:, col], lower=lower, upper=upper)

        return arr

    # def extra_repr(self) -> str:
    #     inplace_str = "inplace=True" if self.inplace else ""
    #     return inplace_str
