"""Core feature."""

# STDLIB
from collections.abc import Mapping, MutableMapping

# THIRD-PARTY
from jax import Array

__all__ = [
    "Array",
    # Parameters
    "FlatParsT",
    "MutableFlatParsT",
    # Data
    "DataT",
    "MutableDataT",
]


# TODO: define these from the stream_ml.core._typing versions

FlatParsT = Mapping[str, Array]
MutableFlatParsT = MutableMapping[str, Array]

DataT = Mapping[str, Array]
MutableDataT = MutableMapping[str, Array]
