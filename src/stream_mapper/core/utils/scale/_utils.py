"""Utilities."""

__all__: tuple[str, ...] = ()

from typing import Protocol


class HasNames(Protocol):
    """Has names."""

    @property
    def names(self) -> tuple[str, ...]:
        """Names."""
        ...


def names_intersect(
    left: HasNames | tuple[str, ...], right: HasNames | tuple[str, ...], /
) -> tuple[str, ...]:
    """Return the intersection of the names of two datasets."""
    return tuple(
        set(left.names if hasattr(left, "names") else left).intersection(
            right.names if hasattr(right, "names") else right
        )
    )
