"""Base background model."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.core.background.base import BackgroundModel as CoreBackgroundModel
from stream_ml.jax.core import ModelBase
from stream_ml.jax.typing import Array

__all__: list[str] = []


@dataclass()
class BackgroundModel(ModelBase, CoreBackgroundModel[Array]):
    """Background Model."""
