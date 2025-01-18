"""Core library for stream membership likelihood, with ML."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from stream_mapper.core.utils.scale._api import ASTYPE_REGISTRY
from stream_mapper.core.utils.scale._standard import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import NDArray


#####################################################################
# StandardScaler


def standard_scaler_astype_numpy(
    scaler: StandardScaler[Any], **kwargs: Any
) -> StandardScaler[NDArray[Any]]:  # type: ignore[type-var]
    """Register the `StandardScaler` class for `numpy.ndarray`."""
    return replace(
        scaler,
        mean=np.asarray(scaler.mean, **kwargs),
        scale=np.asarray(scaler.scale, **kwargs),
        names=scaler.names,
    )


ASTYPE_REGISTRY[(StandardScaler, np.ndarray)] = standard_scaler_astype_numpy  # type: ignore[assignment]
