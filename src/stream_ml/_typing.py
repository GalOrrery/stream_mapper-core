"""Core feature."""

# STDLIB
from collections.abc import Mapping

# THIRD-PARTY
from torch import Tensor

__all__: list[str] = []


ParsT = Mapping[str, Tensor]
DataT = Mapping[str, Tensor]
