"""Core feature."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from stream_ml.core.typing.array import Array, Array_co

__all__: list[str] = []


NNModel = TypeVar("NNModel")
NNModel_co = TypeVar("NNModel_co", covariant=True)


class NNNamespace(Protocol[NNModel_co, Array_co]):
    """Protocol for neural network API namespace."""

    @staticmethod
    def Identity(  # noqa: N802
        *args: Any, **kwargs: Any
    ) -> (
        NNModel_co | Identity[Array_co]
    ):  # NOTE: this should be Intersection[NNModel, Array]
        """Identity."""
        ...


# =============================================================================


class Identity(Protocol[Array]):
    """Protocol for identity."""

    @staticmethod
    def __call__(x: Array) -> Array:
        """Call."""
        ...
