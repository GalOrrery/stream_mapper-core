"""Base background model."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.core.background.base import BackgroundModel as CoreBackgroundModel
from stream_ml.pytorch.base import ModelBase
from stream_ml.pytorch.typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class BackgroundModel(ModelBase, CoreBackgroundModel[Array]):
    """Background Model."""
