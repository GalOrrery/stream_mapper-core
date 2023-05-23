"""Core feature."""

__all__ = [
    "ParamScaler",
    # builtin
    "Identity",
    "StandardLocation",
    "StandardWidth",
    # funcs
    "scale_params",
]

from stream_ml.core.params.scaler._api import ParamScaler
from stream_ml.core.params.scaler._builtin import (
    Identity,
    StandardLocation,
    StandardWidth,
)
from stream_ml.core.params.scaler._utils import scale_params
