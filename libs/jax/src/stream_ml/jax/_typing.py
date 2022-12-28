"""Core feature."""

# STDLIB
from collections.abc import Mapping

# THIRD-PARTY
from jax import Array

__all__ = [
    "Array",
    # Parameters
    "FlatParsT",
    # Data
    "DataT",
]

FlatParsT = Mapping[str, Array]

DataT = Mapping[str, Array]
