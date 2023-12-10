"""Core feature."""

from __future__ import annotations

__all__ = (
    "Array",
    "Array_co",
    "ArrayLike",
    "ArrayNamespace",
    "BoundsT",
    "NNNamespace",
    "NNModel",
    "ParamNameTupleOpts",
)


from collections.abc import Callable
from typing import TypeAlias

from stream_mapper.core.typing._array import Array, Array_co, ArrayLike
from stream_mapper.core.typing._nn import NNModel, NNNamespace
from stream_mapper.core.typing._xp import ArrayNamespace

BoundT: TypeAlias = float | Callable[[float], float]
BoundsT: TypeAlias = tuple[BoundT, BoundT]

ParamNameTupleOpts: TypeAlias = tuple[str] | tuple[str, str]
ParamNameAllOpts: TypeAlias = str | ParamNameTupleOpts

ParamsLikeDict: TypeAlias = dict[str, Array | dict[str, Array]]
