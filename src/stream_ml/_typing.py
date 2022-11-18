"""Core feature."""

# STDLIB
from collections.abc import Mapping

# THIRD-PARTY
from torch import Tensor as Array

__all__: list[str] = ["Array", "DataT", "ParsT"]


ParsT = Mapping[str, Array]
DataT = Mapping[str, Array]
