"""Base background model."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.core.base import ModelBase
from stream_ml.core.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class BackgroundModel(ModelBase[Array]):
    """Background Model."""
