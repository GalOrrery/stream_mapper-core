"""Core feature."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

__all__: list[str] = []


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
