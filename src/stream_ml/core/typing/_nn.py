"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any, Protocol, TypeVar, runtime_checkable

from stream_ml.core.typing._array import Array, Array_co

NNModel = TypeVar("NNModel", bound="NNModelProtocol[Array]")  # type: ignore[valid-type]
NNModel_co = TypeVar("NNModel_co", covariant=True)


class NNNamespace(Protocol[NNModel_co, Array_co]):
    """Protocol for neural network API namespace."""

    @staticmethod
    def Identity(  # noqa: N802
        *args: Any, **kwargs: Any
    ) -> NNModel_co:  # NOTE: this should be Intersection[NNModel, Array]
        """Identity."""
        ...

    @staticmethod
    def Linear(*args: Any, **kwargs: Any) -> NNModel_co:  # noqa: N802
        """Linear."""
        ...

    @staticmethod
    def Sigmoid(*args: Any, **kwargs: Any) -> NNModel_co:  # noqa: N802
        """Sigmoid."""
        ...

    @staticmethod
    def Sequential(*args: Any, **kwargs: Any) -> NNModel_co:  # noqa: N802
        """Sequential."""
        ...


# =============================================================================


@runtime_checkable
class NNModelProtocol(Protocol[Array]):
    """Protocol for Neufal Network Models."""

    @staticmethod
    def __call__(x: Array) -> Array:
        """Call."""
        ...


class IdentityProtocol(NNModelProtocol[Array], Protocol):
    """Protocol for identity."""

    @staticmethod
    def __call__(x: Array) -> Array:
        """Call."""
        ...
