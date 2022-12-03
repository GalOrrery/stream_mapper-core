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
    "ParsT",
    "MutableParsT",
    # Data
    "DataT",
    "MutableDataT",
]


# TODO: define these from the stream_ml.core._typing versions

FlatParsT = Mapping[str, Array]
MutableFlatParsT = MutableMapping[str, Array]

ParsT = Mapping[str, Array | Mapping[str, Array]]
MutableParsT = MutableMapping[str, Array | MutableMapping[str, Array]]

DataT = Mapping[str, Array]
MutableDataT = MutableMapping[str, Array]
