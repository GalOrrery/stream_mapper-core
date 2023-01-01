"""Pytorch type hints."""

# STDLIB
from collections.abc import Mapping

# THIRD-PARTY
from torch import Tensor as Array

__all__ = ["Array", "FlatParsT"]


FlatParsT = Mapping[str, Array]
