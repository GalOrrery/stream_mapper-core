"""Core library for stream membership likelihood, with ML."""


from stream_ml.core import params, prior, utils
from stream_ml.core._base import ModelBase
from stream_ml.core.api import Model
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
    # model classes
    "Model",
    "ModelBase",
    "MixtureModel",
    "IndependentModels",
]


# isort: split
from stream_ml.core import _connect  # noqa: F401
