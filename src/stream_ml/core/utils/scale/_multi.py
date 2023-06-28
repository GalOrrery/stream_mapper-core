"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
import functools
import operator
from typing import TYPE_CHECKING, overload

from stream_ml.core.data import Data
from stream_ml.core.typing import Array
from stream_ml.core.utils.scale._api import DataScaler

if TYPE_CHECKING:
    from stream_ml.core.typing import ArrayNamespace


@dataclass(frozen=True)
class CompoundDataScaler(DataScaler[Array]):
    """Compound scaler."""

    scalers: tuple[DataScaler[Array], ...]

    def __post_init__(self) -> None:
        # Check that all names are unique.
        names = set()
        for scaler in self.scalers:
            for name in scaler.names:
                if name in names:
                    msg = f"duplicate name: {name}"
                    raise ValueError(msg)
                names.add(name)

        self.names: tuple[str, ...]
        object.__setattr__(
            self,
            "names",
            functools.reduce(operator.add, (s.names for s in self.scalers)),
        )

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
        if xp is None:
            msg = "xp must be specified"
            raise ValueError(msg)

        if not isinstance(data, Data):
            data = Data(xp.asarray(data), names=names)

        xd = xp.stack(
            tuple(
                scaler.transform(data, names=names, xp=xp).array
                for scaler in self.scalers
            ),
            1,
        )
        return Data(xd, names=self.names) if isinstance(data, Data) else xd

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
        if xp is None:
            msg = "xp must be specified"
            raise ValueError(msg)

        if not isinstance(data, Data):
            data = Data(xp.asarray(data), names=names)

        xd = xp.stack(
            tuple(
                scaler.inverse_transform(data, names=names, xp=xp).array
                for scaler in self.scalers
            ),
            1,
        )

        return Data(xd, names=self.names) if isinstance(data, Data) else xd

    # ---------------------------------------------------------------

    def __getitem__(self, names: tuple[str, ...]) -> DataScaler[Array]:
        """Get a subset DataScaler with the given names."""
        scalers = tuple(
            scaler[ns]
            for scaler in self.scalers
            if (ns := tuple(n for n in scaler.names if n in names))
        )
        return CompoundDataScaler[Array](scalers) if len(scalers) > 1 else scalers[0]
