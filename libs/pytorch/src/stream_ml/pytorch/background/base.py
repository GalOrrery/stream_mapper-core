"""Base background model."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.pytorch.base import ModelBase

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class BackgroundModel(ModelBase):
    """Background Model."""
