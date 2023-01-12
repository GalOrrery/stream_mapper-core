"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TypeVar

# LOCAL
from stream_ml.core.data import Data as CoreData
from stream_ml.jax.typing import Array

Self = TypeVar("Self", bound="Data[Array]")  # type: ignore[type-arg]


@dataclass(frozen=True)
class Data(CoreData[Array]):
    """Data."""

    def __jax_array__(self) -> Array:
        """Convert to a JAX array."""
        return self.array
