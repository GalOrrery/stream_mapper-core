"""Core feature."""

# STDLIB
from collections.abc import Mapping

# THIRD-PARTY
from jax import Array

__all__ = ["Array", "FlatParsT"]

FlatParsT = Mapping[str, Array]
