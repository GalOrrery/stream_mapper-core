"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stream_ml.core.data import FROM_FORMAT_REGISTRY, Data

__all__: list[str] = []

if TYPE_CHECKING:
    from numpy.typing import NDArray


#####################################################################
# FROM_FORMAT


def _from_structured_array(array: NDArray[Any], /) -> Data[NDArray[Any]]:
    """Create a `Data` instance from a structured numpy array.

    Requires :mod:`numpy` to be installed.

    Parameters
    ----------
    array : ndarray
        The structured array.

    Returns
    -------
    Data
        The data instance.
    """
    from numpy.lib.recfunctions import structured_to_unstructured

    if not isinstance(array.dtype.names, tuple):
        msg = "The array must be structured."
        raise TypeError(msg)

    return Data(structured_to_unstructured(array), names=array.dtype.names)


try:
    import numpy as np  # noqa: F401
except ImportError:
    FROM_FORMAT_REGISTRY["numpy.structured"] = _from_structured_array
