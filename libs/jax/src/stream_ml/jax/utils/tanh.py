"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

# THIRD-PARTY
import flax.linen as nn

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.jax._typing import Array

__all__: list[str] = []


class Tanh(nn.Module):  # type: ignore[misc]
    """Tanh activation function as a Module."""

    def setup(self) -> None:
        """Setup."""
        pass

    def __call__(self, x: Array) -> Array:
        """Call."""
        return nn.tanh(x)
