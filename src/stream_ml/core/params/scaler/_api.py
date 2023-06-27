"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.typing import ArrayNamespace
    from stream_ml.core.utils.scale import DataScaler

T = TypeVar("T", bound=str | EllipsisType)
ParamScalerT = TypeVar("ParamScalerT", bound="ParamScaler[Array]")  # type: ignore[valid-type]  # noqa: E501


@runtime_checkable
class ParamScaler(Protocol[Array]):
    """Protocol for parameter scalers."""

    def transform(self, data: Array | float, /) -> Array:
        """Transform the data."""
        ...

    def inverse_transform(self, data: Array, /) -> Array:
        """Inverse transform the data."""
        ...

    @classmethod
    def from_data_scaler(
        cls: type[ParamScalerT],
        scaler: DataScaler,
        /,
        name: str,
        *,
        xp: ArrayNamespace[Array] | None = None,
    ) -> ParamScalerT:
        """Construct from ``DataScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.DataScaler`
            The scaler object.
        name : str
            The name of the scaling to extract.

        xp : array namespace, optional keyword-only
            The array namespace to use, by default None.

        Returns
        -------
        ``ParamScaler``
        """
        ...
