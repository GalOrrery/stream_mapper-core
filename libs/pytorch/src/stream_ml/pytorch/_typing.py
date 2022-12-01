"""Core feature."""

# STDLIB

# THIRD-PARTY
from torch import Tensor as Array

# LOCAL
from stream_ml.core._typing import DataT as _DataT
from stream_ml.core._typing import ParsT as _ParsT

__all__: list[str] = ["Array", "DataT", "ParsT"]


ParsT = _ParsT[Array]
DataT = _DataT[Array]
