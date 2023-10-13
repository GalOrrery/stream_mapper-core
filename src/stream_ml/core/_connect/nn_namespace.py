"""Connect array namespace to NN namespace."""

__all__: tuple[str, ...] = ()

from typing import Protocol, cast

from stream_ml.core.typing import Array, ArrayNamespace, NNModel, NNNamespace


class NNNamespaceMap(Protocol):
    """Protocol for mapping array namespaces to NN namespaces."""

    def __getitem__(self, key: ArrayNamespace[Array]) -> NNNamespace[NNModel, Array]:
        """Get item."""
        ...

    def __setitem__(
        self, key: ArrayNamespace[Array], value: NNNamespace[NNModel, Array]
    ) -> None:
        """Set item."""
        ...


NN_NAMESPACE = cast(NNNamespaceMap, {})
