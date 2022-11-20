"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

if TYPE_CHECKING:
    # LOCAL
    from stream_ml._typing import Array

__all__: list[str] = []


_halfln2pi = 0.5 * xp.log(xp.asarray([2]) * xp.pi)


def log_of_normal(
    x: Array,
    mu: Array,
    sigma: Array,
) -> Array:
    """Log of Gaussian distribution.

    Parameters
    ----------
    x : Array
        X.
    mu : Array
        Mu.
    sigma : Array
        Sigma.
    """
    return -0.5 * ((x - mu) / sigma) ** 2 - xp.log(sigma) - _halfln2pi
