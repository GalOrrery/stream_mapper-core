"""Core feature."""

from __future__ import annotations

from typing import TypeAlias

from .array import Array, ArrayLike, ArrayNamespace
from .nn import NNModel, NNNamespace

__all__ = ["Array", "ArrayLike", "ArrayNamespace", "BoundsT", "NNNamespace", "NNModel"]


BoundsT: TypeAlias = tuple[float, float]
