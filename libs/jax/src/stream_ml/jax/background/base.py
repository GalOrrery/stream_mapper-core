"""Base background model."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.jax.core import ModelBase

__all__: list[str] = []


@dataclass()
class BackgroundModel(ModelBase):
    """Background Model."""
