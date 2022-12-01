"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array

__all__: list[str] = []


_halfln2pi = 0.5 * xp.log(xp.asarray([2]) * xp.pi)


def norm_logpdf(x: Array, *, mu: Array, sigma: Array, amp: Array) -> Array:
    """Log of Gaussian distribution.

    Parameters
    ----------
    x : Array
        X.
    mu : Array
        Mu.
    sigma : Array
        Sigma.
    amp : Array
        Amplitude.

    Returns
    -------
    Array
    """
    return (
        xp.log(xp.clamp(amp, min=0))
        + -0.5 * ((x - mu) / sigma) ** 2
        - xp.log(xp.clamp(sigma, min=0))
        - _halfln2pi
    )


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
    return xp.sigmoid(x) * (upper - lower) + lower
