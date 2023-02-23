"""Core library for stream membership likelihood, with ML."""


from stream_ml.core import params, prior, utils
from stream_ml.core.data import Data
from stream_ml.core.multi.independent import IndependentModels
from stream_ml.core.multi.mixture import MixtureModel

__all__ = [
    # classes
    "Data",
    # modules
    "utils",
    "params",
    "prior",
    # model classes
    "MixtureModel",
    "IndependentModels",
]


# isort: split
from . import _connect  # noqa: F401
