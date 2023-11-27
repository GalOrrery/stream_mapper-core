"""Built-in background models."""

from __future__ import annotations

__all__ = [
    "HardCutoffMassFunction",
    "StepwiseMassFunction",
    "StreamMassFunction",
    "UniformStreamMassFunction",
]

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from stream_ml.core.utils.compat import array_at

if TYPE_CHECKING:
    from stream_ml.core import Data
    from stream_ml.core.typing import Array, ArrayNamespace

# =============================================================================
# Cluster Mass Function


@runtime_checkable
class StreamMassFunction(Protocol):
    """Stream Mass Function.

    Must be parametrized by gamma [0, 1], the normalized mass over the range of the
    isochrone.

    Returns the log-probability that stars of that mass (gamma) are in the
    population modeled by the isochrone.
    """

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        r"""Log-probability of stars at position 'x' having mass 'gamma'.

        Parameters
        ----------
        gamma : Array[(F,))]
            The mass of the stars, normalized to [0, 1] over the range of the
            isochrone.
        x : Data[Array[(N,)]]
            The independent data. Normally this is :math:`\phi_1`.

        xp : ArrayNamespace[Array], keyword-only
            The array namespace.

        Returns
        -------
        Array[(N, F)]
        """
        ...


@dataclass(frozen=True)
class UniformStreamMassFunction(StreamMassFunction):
    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        return xp.zeros((len(x), len(gamma)))


@dataclass(frozen=True)
class HardCutoffMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    low: float = 0
    upper: float = 1

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((len(x), len(gamma)), -xp.inf)
        return array_at(
            out,
            (slice(None), (gamma >= self.low) & (gamma <= self.upper)),
            inplace=True,
        ).set(0)


@dataclass(frozen=True)
class StepwiseMassFunction(StreamMassFunction):
    """Hard Cutoff IMF."""

    boundaries: tuple[float, ...]  # (B + 1,)
    log_probs: tuple[float, ...]  # (B,)

    def __call__(
        self, gamma: Array, x: Data[Array], *, xp: ArrayNamespace[Array]
    ) -> Array:
        out = xp.full((len(x), len(gamma)), -xp.inf)
        for (lw, up), lp in zip(pairwise(self.boundaries), self.log_probs, strict=True):
            out = array_at(
                out,
                (slice(None), (gamma >= lw) & (gamma < up)),
                inplace=True,
            ).set(lp)
        return out
