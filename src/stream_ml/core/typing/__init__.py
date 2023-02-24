"""Core feature."""

from __future__ import annotations

from typing import TypeAlias

from stream_ml.core.typing.array import Array, ArrayLike, ArrayNamespace
from stream_ml.core.typing.nn import NNModel, NNNamespace

__all__ = ["Array", "ArrayLike", "ArrayNamespace", "BoundsT", "NNNamespace", "NNModel"]


BoundsT: TypeAlias = tuple[float, float]
