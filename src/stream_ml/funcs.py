"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

__all__: list[str] = []


_halfln2pi = 0.5 * xp.log(xp.tensor([2]) * xp.pi)


def log_of_normal(
    x: Tensor,
    mu: Tensor,
    sigma: Tensor,
) -> Tensor:
    """Log of Gaussian distribution.

    Parameters
    ----------
    x : Tensor
        X.
    mu : Tensor
        Mu.
    sigma : Tensor
        Sigma.
    """
    return -0.5 * ((x - mu) / sigma) ** 2 - xp.log(sigma) - _halfln2pi
