"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import flax.linen as nn

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.jax._typing import Array

__all__: list[str] = []


def sigmoid(x: Array, /, lower: Array | float = 0, upper: Array | float = 1) -> Array:
    """Sigmoid function.

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
    return nn.sigmoid(x) * (upper - lower) + lower


class ColumnarScaledSigmoid(nn.Module):  # type: ignore[misc]
    """Tanh activation function as a Module."""

    columns: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]

    def setup(self) -> None:
        """Setup."""
        if len(self.columns) != len(self.bounds):
            raise ValueError("columns and bounds must be the same length")

    def __call__(self, arr: Array) -> Array:
        """Call."""
        for col, (lower, upper) in zip(self.columns, self.bounds):
            arr = arr.at[:, col].set(sigmoid(arr[:, col], lower=lower, upper=upper))
        return arr
