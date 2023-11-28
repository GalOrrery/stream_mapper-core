"""connect to copy."""

__all__: tuple[str, ...] = ()

from typing import Any, TypeVar, cast

import numpy as np
import numpy.typing as npt

from stream_mapper.core.typing import ArrayNamespace
from stream_mapper.core.utils.compat import copy, get_namespace

T = TypeVar("T", bound=np.generic)


@copy.register(np.ndarray)
def _copy_numpy(array: npt.NDArray[T], /) -> npt.NDArray[T]:  # type: ignore[misc]
    """Copy numpy array."""
    return np.copy(array)


@get_namespace.register(np.ndarray)  # type: ignore[misc]
def _get_namespace_numpy(
    array: npt.NDArray[Any], /
) -> ArrayNamespace[npt.NDArray[Any]]:
    """Get numpy namespace."""
    return cast("ArrayNamespace[npt.NDArray[Any]]", np)
