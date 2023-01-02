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


def scaled_sigmoid(
    x: Array, /, lower: Array | float = 0, upper: Array | float = 1
) -> Array:
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
