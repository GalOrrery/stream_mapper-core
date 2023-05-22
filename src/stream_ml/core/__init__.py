"""Core library for stream membership likelihood, with ML."""


from stream_ml.core import builtin, params, prior, utils
from stream_ml.core._api import Model
from stream_ml.core._base import ModelBase, NNField
from stream_ml.core.data import Data
from stream_ml.core.multi._independent import IndependentModels
from stream_ml.core.multi._mixture import MixtureModel

__all__ = [
    # classes
    "Data",
    # modules
    "utils",
    "params",
    "prior",
    "builtin",
    # model classes
    "Model",
    "ModelBase",
    "MixtureModel",
    "IndependentModels",
    # model related
    "NNField",
]


# isort: split
from stream_ml.core import _connect  # noqa: F401
