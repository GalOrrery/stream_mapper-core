"""Establishing the probabilities API, largely free of any implementation."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, Protocol, SupportsIndex

from stream_ml.core._connect.nn_namespace import NN_NAMESPACE
from stream_ml.core._connect.xp_namespace import XP_NAMESPACE, XP_NAMESPACE_REVERSE
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

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        """Reduce."""
        reduced = super().__reduce_ex__(protocol)
        if isinstance(reduced, str):
            return reduced

        reduced[2]["array_namespace"] = (
            XP_NAMESPACE_REVERSE[self.array_namespace]
            if self.array_namespace in XP_NAMESPACE_REVERSE
            else self.array_namespace
        )
        return reduced

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state."""
        try:
            super().__setstate__(state)  # type: ignore[misc]
        except AttributeError:
            self.__dict__.update(state)
        object.__setattr__(self, "array_namespace", XP_NAMESPACE[self.array_namespace])


class SupportsXPNN(SupportsXP[Array], Protocol[Array, NNModel]):
    """Protocol for objects that support array and NN namespaces."""

    _nn_namespace_: NNNamespace[NNModel, Array]

    @property
    def xpnn(self) -> NNNamespace[NNModel, Array]:
        """NN namespace."""
        return self._nn_namespace_

    def __reduce_ex__(self, protocol: Any) -> str | tuple[Any, ...]:
        """Reduce."""
        reduced = super().__reduce_ex__(protocol)
        if isinstance(reduced, str):
            return reduced

        reduced[2].pop("_nn_namespace_", None)
        return reduced

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state."""
        super().__setstate__(state)
        object.__setattr__(self, "_nn_namespace_", NN_NAMESPACE[self.array_namespace])


class HasName(Protocol):
    """Protocol for objects that have a name."""

    name: str | None
