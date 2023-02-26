"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from numpy.lib.recfunctions import (
    structured_to_unstructured,
    unstructured_to_structured,
)

from stream_ml.core.data import FROM_FORMAT_REGISTRY, TO_FORMAT_REGISTRY, Data

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.table import Table
    from numpy.typing import NDArray


#####################################################################
# NUMPY


def _from_structured_array(array: NDArray[Any], /, **kwargs: Any) -> Data[NDArray[Any]]:
    """Create a `Data` instance from a structured numpy array.

    Requires :mod:`numpy` to be installed.

    Parameters
    ----------
    array : ndarray
        The structured array.
    **kwargs : Any
        Additional keyword arguments. Possible values are:

        - names : tuple[str, ...] | None
            The names of the columns to keep. Default (`None`) is to keep all
            columns.
        - renamer : dict[str, str] | None
            A dictionary of column names to rename. Default (`None`) is to not
            rename any.

    Returns
    -------
    Data
        The data instance.
    """
    if not isinstance(array.dtype.names, tuple):
        msg = "The array must be structured."
        raise TypeError(msg)

    names = _names if (_names := kwargs.get("names")) is not None else array.dtype.names
    renamer = _renamer if (_renamer := kwargs.get("renamer")) is not None else {}

    return Data(
        structured_to_unstructured(array[list(names)]),  # unstructured array selection
        names=tuple(renamer.get(n, n) for n in names),  # column (re)names
    )


def _to_format_structured_array(data: Data[NDArray[Any]], /) -> NDArray[Any]:
    """Convert the data to a structured numpy array.

    Requires :mod:`numpy` to be installed.

    Parameters
    ----------
    data : Data
        The data.

    Returns
    -------
    ndarray
        The structured array.
    """
    return cast(
        "NDArray[Any]", unstructured_to_structured(data.array, names=data.names)
    )


try:
    import numpy as np
except ImportError:
    pass
else:
    FROM_FORMAT_REGISTRY["numpy.structured"] = _from_structured_array
    TO_FORMAT_REGISTRY[(np.ndarray, np.ndarray)] = _to_format_structured_array


#####################################################################
# ASTROPY


def _from_astropy_table(table: Table, /, **kwargs: Any) -> Data[NDArray[Any]]:
    """Create a `Data` instance from an `astropy.table.Table`.

    Requires :mod:`astropy` to be installed.

    Parameters
    ----------
    table : astropy.table.Table
        The table.
    **kwargs : Any
        Additional keyword arguments. Possible values are:

        - keep_byteorder : bool
            Whether to keep the byte order of the array. Default is `False`.
        - names : tuple[str, ...] | None
            The names of the columns to keep. Default (`None`) is to keep all
            columns.
        - renamer : dict[str, str] | None
            A dictionary of column names to rename. Default (`None`) is to not
            rename any.

    Returns
    -------
    Data
        The data instance.
    """
    return Data.from_format(
        table.as_array(
            keep_byteorder=kwargs.pop("keep_byteorder", False),
            names=kwargs.pop("names", None),
        ),
        fmt="numpy.structured",
        **kwargs,
    )


try:
    import astropy  # noqa: F401
except ImportError:
    pass
else:
    FROM_FORMAT_REGISTRY["astropy.table"] = _from_astropy_table
