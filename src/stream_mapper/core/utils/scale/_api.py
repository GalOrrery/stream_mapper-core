"""Core feature."""

from __future__ import annotations

__all__: list[str] = ["ASTYPE_REGISTRY"]
# ASTYPE_REGISTRY is public in this private module.

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, overload

from stream_mapper.core.typing import Array

if TYPE_CHECKING:
    from typing_extensions import Self

    from stream_mapper.core._data import Data
    from stream_mapper.core.typing import ArrayLike, ArrayNamespace

DS = TypeVar("DS", bound="DataScaler")  # type: ignore[type-arg]


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

    def __getitem__(self: Self, names: str | tuple[str, ...]) -> Self:
        """Get a subset DataScaler with the given names."""
        ...

    # ===============================================================

    def astype(self: DS[Any], fmt: type[Array], /, **kwargs: Any) -> DS[Array]:  # type: ignore[valid-type]
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
        return cast("DS[Array]", ASTYPE_REGISTRY[(type(self), fmt)](self, **kwargs))  # type: ignore[valid-type]


###############################################################################
# HOOKS


class AsTypeConverter(Protocol):
    """ASTYPE_REGISTRY protocol."""

    def __call__(self, obj: DataScaler[Any], /, **kwargs: Any) -> DataScaler[ArrayLike]:
        ...


ASTYPE_REGISTRY: dict[tuple[type, type], AsTypeConverter] = {}
