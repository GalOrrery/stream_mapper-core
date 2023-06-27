"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

from stream_ml.core.params.scaler._api import ParamScaler
from stream_ml.core.typing import Array
from stream_ml.core.utils.scale import StandardScaler

if TYPE_CHECKING:
    from stream_ml.core.typing import ArrayNamespace
    from stream_ml.core.utils.scale import DataScaler

T = TypeVar("T", bound=str | EllipsisType)
ParamScalerT = TypeVar("ParamScalerT", bound="ParamScaler[Array]")  # type: ignore[valid-type]  # noqa: E501


@dataclass(frozen=True, slots=True)
class Identity(ParamScaler[Array]):
    """Identity scaler."""

    def transform(self, data: Array | float, /) -> Array | float:  # type: ignore[override]  # noqa: E501
        """Transform the data."""
        return data

    def inverse_transform(self, data: Array, /) -> Array:
        """Inverse transform the data."""
        return data

    @classmethod
    def from_data_scaler(
        cls: type[ParamScalerT],
        scaler: DataScaler,  # noqa: ARG003
        /,
        name: str,  # noqa: ARG003
        *,
        xp: ArrayNamespace[Array] | None = None,  # noqa: ARG003
    ) -> ParamScalerT:
        """Construct from ``DataScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.DataScaler`
            The scaler object. Not used.
        name : str
            The name of the scaling to extract. Not used.

        xp : array namespace, optional keyword-only
            The array namespace to use, by default None.

        Returns
        -------
        ``Identity``
        """
        return cls()


@dataclass(frozen=True, slots=True)
class StandardLocation(ParamScaler[Array]):
    """Standard scaler for a location, which need mean and scale."""

    mean: Array | float
    scale: Array | float

    def transform(self, data: Array | float, /) -> Array:
        """Transform the data."""
        return (data - self.mean) / self.scale  # type: ignore[return-value]

    def inverse_transform(self, data: Array, /) -> Array:
        """Inverse transform the data."""
        return data * self.scale + self.mean

    @classmethod
    def from_data_scaler(
        cls: type[StandardLocation[Array]],
        scaler: DataScaler,
        /,
        name: str,
        *,
        xp: ArrayNamespace[Array] | None = None,  # noqa: ARG003
    ) -> StandardLocation[Array]:
        """Construct from ``StandardScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.DataScaler`
            The scaler object. Must be a `stream_ml.core.utils.scale.StandardScaler`
        name : str
            The name of the scaling to extract.

        xp : array namespace, optional keyword-only
            The array namespace to use, by default None.

        Returns
        -------
        ``StandardLocation``
        """
        if not isinstance(scaler, StandardScaler):
            msg = f"scaler must be a <StandardScaler>, not {type(scaler)}"
            raise TypeError(msg)

        i = scaler.names.index(name)
        return cls(mean=scaler.mean[i], scale=scaler.scale[i])


@dataclass(frozen=True, slots=True)
class StandardWidth(ParamScaler[Array]):
    """Standard scaler for a width, which needs only the scale."""

    scale: Array | float

    def transform(self, width: Array | float, /) -> Array:
        """Transform the data."""
        return width / self.scale  # type: ignore[return-value]

    def inverse_transform(self, data: Array, /) -> Array:
        """Inverse transform the data."""
        return data * self.scale

    @classmethod
    def from_data_scaler(
        cls: type[StandardWidth[Array]],
        scaler: DataScaler,
        /,
        name: str,
        *,
        xp: ArrayNamespace[Array] | None = None,  # noqa: ARG003
    ) -> StandardWidth[Array]:
        """Construct from ``StandardScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.StandardScaler`
            The scaler object.
        name : str
            The name of the scaling to extract.

        xp : array namespace, optional keyword-only
            The array namespace to use, by default None.

        Returns
        -------
        ``StandardWidth``
        """
        if not isinstance(scaler, StandardScaler):
            msg = f"scaler must be a <StandardScaler>, not {type(scaler)}"
            raise TypeError(msg)

        return cls(scale=scaler.scale[scaler.names.index(name)])


@dataclass(frozen=True, slots=True)
class StandardLnWidth(ParamScaler[Array]):
    """Standard scaler for a log-width, which needs only the scale."""

    ln_scale: Array | float

    def transform(self, ln_width: Array | float, /) -> Array | float:  # type: ignore[override]  # noqa: E501
        """Transform the ln_width."""
        return ln_width - self.ln_scale

    def inverse_transform(self, ln_width: Array, /) -> Array:
        """Inverse transform the ln_width."""
        return ln_width + self.ln_scale

    @classmethod
    def from_data_scaler(
        cls: type[StandardLnWidth[Array]],
        scaler: DataScaler,
        /,
        name: str,
        *,
        xp: ArrayNamespace[Array] | None = None,
    ) -> StandardLnWidth[Array]:
        """Construct from ``StandardScaler`` object.

        Parameters
        ----------
        scaler : `stream_ml.core.utils.scale.StandardScaler`
            The scaler object.
        name : str
            The name of the scaling to extract.

        xp : array namespace, optional keyword-only
            The array namespace to use, by default None.

        Returns
        -------
        ``StandardLnWidth``
        """
        if not isinstance(scaler, StandardScaler):
            msg = f"scaler must be a <StandardScaler>, not {type(scaler)}"
            raise TypeError(msg)
        if xp is None:
            msg = "xp must be provided"
            raise ValueError(msg)

        return cls(ln_scale=xp.log(scaler.scale[scaler.names.index(name)]))
