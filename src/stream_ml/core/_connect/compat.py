"""connect to copy."""

__all__: list[str] = []

from typing import TypeVar

import numpy as np
import numpy.typing as npt

from stream_ml.core.utils.compat import copy

T = TypeVar("T", bound=np.generic)


@copy.register(np.ndarray)
def _copy_numpy(array: npt.NDArray[T], /) -> npt.NDArray[T]:  # type: ignore[misc]
    """Copy numpy array."""
    return np.copy(array)
