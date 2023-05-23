"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class DataScaler(Protocol):
    """Data scaler protocol."""

    names: tuple[str, ...]

    def transform(self: DataScaler, data: T, /, names: tuple[str, ...]) -> T:
        """Scale features of X according to feature_range."""
        ...

    def inverse_transform(
        self: DataScaler, data: T, /, names: tuple[str, ...], **kwargs: Any
    ) -> T:
        """Scale features of X according to feature_range."""
        ...
