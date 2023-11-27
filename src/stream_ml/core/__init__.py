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

from stream_ml.core import builtin, params, prior
from stream_ml.core._core.base import ModelBase
from stream_ml.core._core.field import NNField
from stream_ml.core._core.model_api import Model as ModelAPI
from stream_ml.core._data import Data
from stream_ml.core._multi.bases import ModelsBase
from stream_ml.core._multi.independent import IndependentModels
from stream_ml.core._multi.mixture import MixtureModel
from stream_ml.core.params._values import Params
from stream_ml.core.setup_package import BACKGROUND_KEY, WEIGHT_NAME

# isort: split
from stream_ml.core import _connect  # noqa: F401
