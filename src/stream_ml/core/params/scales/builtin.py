"""Core feature."""

from __future__ import annotations

from dataclasses import dataclass
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from stream_ml.core.typing import Array
from stream_ml.core.utils.scale import StandardScaler

if TYPE_CHECKING:
    from stream_ml.core.utils.scale import DataScaler

T = TypeVar("T", bound=str | EllipsisType)
ParamScalerT = TypeVar("ParamScalerT", bound="ParamScaler[Array]")  # type: ignore[valid-type]  # noqa: E501

__all__: list[str] = []


@runtime_checkable
class ParamScaler(Protocol[Array]):
    """Protocol for parameter scalers."""

    def transform(self, data: Array | float) -> Array | float:
        """Transform the data."""
        ...

    def inverse_transform(self, data: Array) -> Array:
        """Inverse transform the data."""
        ...

    @classmethod
    def from_data_scaler(
        cls: type[ParamScalerT], scaler: DataScaler, name: str
    ) -> ParamScalerT:
        """Construct from ``DataScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.DataScaler`
            The scaler object.
        name : str
            The name of the scaling to extract.

        Returns
        -------
        ``ParamScaler``
        """
        ...


@dataclass(frozen=True)
class Identity(ParamScaler[Array]):
    """Identity scaler."""

    def transform(self, data: Array | float) -> Array | float:
        """Transform the data."""
        return data

    def inverse_transform(self, data: Array) -> Array:
        """Inverse transform the data."""
        return data

    @classmethod
    def from_data_scaler(
        cls: type[ParamScalerT], scaler: DataScaler, name: str  # noqa: ARG003
    ) -> ParamScalerT:
        """Construct from ``DataScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.DataScaler`
            The scaler object. Not used.
        name : str
            The name of the scaling to extract. Not used.

        Returns
        -------
        ``Identity``
        """
        return cls()


@dataclass(frozen=True)
class StandardLocation(ParamScaler[Array]):
    """Standard scaler for a location, which need mean and scale."""

    mean: Array
    scale: Array

    def transform(self, data: Array | float) -> Array | float:
        """Transform the data."""
        return (data - self.mean) / self.scale

    def inverse_transform(self, data: Array) -> Array:
        """Inverse transform the data."""
        return data * self.scale + self.mean

    @classmethod
    def from_data_scaler(
        cls: type[StandardLocation[Array]], scaler: DataScaler, name: str
    ) -> StandardLocation[Array]:
        """Construct from ``StandardScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.DataScaler`
            The scaler object. Must be a `stream_ml.core.utils.scale.StandardScaler`
        name : str
            The name of the scaling to extract.

        Returns
        -------
        ``StandardLocation``
        """
        if not isinstance(scaler, StandardScaler):
            msg = f"scaler must be a <StandardScaler>, not {type(scaler)}"
            raise TypeError(msg)

        i = scaler.names.index(name)
        return cls(mean=scaler.mean[i], scale=scaler.scale[i])


@dataclass(frozen=True)
class StandardWidth(ParamScaler[Array]):
    """Standard scaler for a width, which needs only the scale."""

    scale: Array

    def transform(self, data: Array | float) -> Array | float:
        """Transform the data."""
        return data / self.scale

    def inverse_transform(self, data: Array) -> Array:
        """Inverse transform the data."""
        return data * self.scale

    @classmethod
    def from_data_scaler(
        cls: type[StandardWidth[Array]], scaler: DataScaler, name: str
    ) -> StandardWidth[Array]:
        """Construct from ``StandardScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.StandardScaler`
            The scaler object.
        name : str
            The name of the scaling to extract.

        Returns
        -------
        ``StandardWidth``
        """
        if not isinstance(scaler, StandardScaler):
            msg = f"scaler must be a <StandardScaler>, not {type(scaler)}"
            raise TypeError(msg)

        return cls(scale=scaler.scale[scaler.names.index(name)])
