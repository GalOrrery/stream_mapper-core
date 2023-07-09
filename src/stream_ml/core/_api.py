"""Establishing the probabilities API, largely free of any implementation."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Protocol

from stream_ml.core.typing import Array, ArrayNamespace, NNModel

if TYPE_CHECKING:
    from stream_ml.core.typing import NNNamespace


class SupportsXP(Protocol[Array]):
    """Protocol for objects that support array namespaces."""

    array_namespace: ArrayNamespace[Array]

    @property
    def xp(self) -> ArrayNamespace[Array]:
        """Array namespace."""
        return self.array_namespace


class SupportsXPNN(SupportsXP[Array], Protocol[Array, NNModel]):
    """Protocol for objects that support array and NN namespaces."""

    _nn_namespace_: NNNamespace[NNModel, Array]

    @property
    def xpnn(self) -> NNNamespace[NNModel, Array]:
        """NN namespace."""
        return self._nn_namespace_


class HasName(Protocol):
    """Protocol for objects that have a name."""

    name: str | None
