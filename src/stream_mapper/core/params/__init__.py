"""Parameters."""

__all__ = (
    # modules
    "bounds",
    "scaler",
    # parameters
    "ModelParameter",  # A single parameter description
    "ModelParameters",  # collection thereof, generally on a model
    "ModelParametersField",  # for get/set of ModelParameters
    # values
    "Params",  # The values of the parameters
    "freeze_params",
    "unfreeze_params",
    "set_param",
    "add_prefix",
    # scaler api
    "ParamScaler",
    "scale_params",
)

from stream_mapper.core.params import bounds, scaler
from stream_mapper.core.params._collection import ModelParameters
from stream_mapper.core.params._core import ModelParameter
from stream_mapper.core.params._field import ModelParametersField
from stream_mapper.core.params._values import (
    Params,
    add_prefix,
    freeze_params,
    set_param,
    unfreeze_params,
)
from stream_mapper.core.params.scaler._api import ParamScaler
from stream_mapper.core.params.scaler._utils import scale_params
