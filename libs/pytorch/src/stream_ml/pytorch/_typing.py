"""Pytorch type hints."""

# STDLIB
from collections.abc import Mapping

# THIRD-PARTY
from torch import Tensor as Array

__all__ = [
    "Array",
    # Parameters
    "FlatParsT",
    # Data
    "DataT",
]


FlatParsT = Mapping[str, Array]

DataT = Mapping[str, Array]
