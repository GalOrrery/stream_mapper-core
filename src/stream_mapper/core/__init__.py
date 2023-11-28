"""Core library for stream membership likelihood, with ML."""

__all__ = (
    # classes
    "Data",
    "Params",
    # modules
    "params",
    "prior",
    "builtin",
    # model classes
    "ModelAPI",
    "ModelBase",
    "ModelsBase",
    "MixtureModel",
    "IndependentModels",
    # model related
    "NNField",
    # misc
    "WEIGHT_NAME",
    "BACKGROUND_KEY",
)

from stream_mapper.core import builtin, params, prior
from stream_mapper.core._core.base import ModelBase
from stream_mapper.core._core.field import NNField
from stream_mapper.core._core.model_api import Model as ModelAPI
from stream_mapper.core._data import Data
from stream_mapper.core._multi.bases import ModelsBase
from stream_mapper.core._multi.independent import IndependentModels
from stream_mapper.core._multi.mixture import MixtureModel
from stream_mapper.core.params._values import Params
from stream_mapper.core.setup_package import BACKGROUND_KEY, WEIGHT_NAME

# isort: split
from stream_mapper.core import _connect  # noqa: F401
