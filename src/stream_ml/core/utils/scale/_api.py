"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, Protocol, cast, overload

from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from stream_ml.core._data import Data
    from stream_ml.core.typing import ArrayLike, ArrayNamespace


class DataScaler(Protocol[Array]):
    """Data scaler protocol."""

    names: tuple[str, ...]

    # ---------------------------------------------------------------

    @overload
    def transform(
        self,
        data: Data[Array] | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array]:
        ...

    @overload
    def transform(
        self,
        data: Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Array:
        ...

    def transform(
        self,
        data: Data[Array] | Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array] | Array:
        """Scale features of X according to feature_range."""
        ...

    # ---------------------------------------------------------------

    @overload
    def inverse_transform(
        self,
        data: Data[Array],
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array]:
        ...

    @overload
    def inverse_transform(
        self,
        data: Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Array:
        ...

    def inverse_transform(
        self,
        data: Data[Array] | Array | float,
        /,
        names: tuple[str, ...],
        *,
        xp: ArrayNamespace[Array] | None,
    ) -> Data[Array] | Array:
        """Scale features of X according to feature_range."""
        ...

    # ---------------------------------------------------------------

    def __getitem__(
        self: DataScaler[Array], names: str | tuple[str, ...]
    ) -> DataScaler[Array]:
        """Get a subset DataScaler with the given names."""
        ...

    # ===============================================================

    def astype(self, fmt: type[Array], /, **kwargs: Any) -> DataScaler[Array]:
        """Convert the data to a different format.

        Parameters
        ----------
        fmt : type
            The format to convert to.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        DataScaler
        """
        return cast(
            "DataScaler[Array]", ASTYPE_REGISTRY[(type(self), fmt)](self, **kwargs)
        )


###############################################################################
# HOOKS


class AsTypeConverter(Protocol):
    """ASTYPE_REGISTRY protocol."""

    def __call__(self, obj: DataScaler[Any], /, **kwargs: Any) -> DataScaler[ArrayLike]:
        ...


ASTYPE_REGISTRY: dict[tuple[type, type], AsTypeConverter] = {}
