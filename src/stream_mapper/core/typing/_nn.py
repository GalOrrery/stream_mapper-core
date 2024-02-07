"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from typing import Protocol, TypeVar, runtime_checkable

from stream_mapper.core.typing._array import Array, Array_co

NNModel = TypeVar("NNModel", bound="NNModelProtocol[Array]")  # type: ignore[valid-type]
NNModel_co = TypeVar("NNModel_co", covariant=True)


class NNNamespace(Protocol[NNModel_co, Array_co]):
    """Protocol for neural network API namespace."""


# =============================================================================


@runtime_checkable
class NNModelProtocol(Protocol[Array]):
    """Protocol for Neural Network Models."""

    @staticmethod
    def __call__(x: Array) -> Array: ...
