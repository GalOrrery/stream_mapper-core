"""Core feature."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from collections.abc import Mapping
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
    overload,
)

from stream_ml.core.utils.frozen_dict import FrozenDict

if TYPE_CHECKING:
    from collections.abc import Iterable

    from stream_ml.core.params.names import ParamNames

Object = TypeVar("Object")
T = TypeVar("T", bound=str | EllipsisType)

__all__: list[str] = []


##############################################################################


class ParamXsBase(
    FrozenDict[str | T, Object | FrozenDict[str, Object]], metaclass=ABCMeta
):
    """Base class for parameterized objects."""

    _Object: type[Object]

    @staticmethod
    @abstractmethod
    def _prepare_freeze(
        xs: dict[str | T, Object | None | Mapping[str, Object | None]], /
    ) -> dict[str | T, Object | FrozenDict[str, Object]]:
        ...

    def __init__(
        self, m: Any = (), /, *, __unsafe_skip_copy__: bool = False, **kwargs: Any
    ) -> None:
        if __unsafe_skip_copy__ and not kwargs:
            super().__init__(m, __unsafe_skip_copy__=True)

        # Initialize, with validation.
        # TODO: not cast to dict if already a ParamBounds
        pb: dict[
            str | T,
            Object | FrozenDict[str, Object],
        ]
        pb = self._prepare_freeze(dict(m, **kwargs))

        super().__init__(pb, __unsafe_skip_copy__=True)

    # =========================================================================
    # Mapping

    @overload
    def __getitem__(self, key: str | T) -> Object | FrozenDict[str, Object]:
        ...

    @overload
    def __getitem__(self, key: tuple[str] | tuple[str, str]) -> Object:  # Flat keys
        ...

    def __getitem__(
        self, key: str | T | tuple[str] | tuple[str, str]
    ) -> Object | FrozenDict[str, Object]:
        if isinstance(key, tuple):
            if len(key) == 1:  # e.g. ("weight",)
                value = super().__getitem__(key[0])
                if not isinstance(value, self._Object):
                    raise KeyError(key)
            else:  # e.g. ("phi2", "mu")
                key = cast("tuple[str, str]", key)  # TODO: remove cast
                v = super().__getitem__(key[0])
                if not isinstance(v, FrozenDict):
                    raise KeyError(key)
                value = v[key[1]]
        else:  # e.g. "weight"
            value = super().__getitem__(key)
        return value

    @overload
    def __contains__(self, o: str | T, /) -> bool:
        ...

    @overload
    def __contains__(self, o: tuple[str] | tuple[str, str], /) -> bool:
        ...

    @overload
    def __contains__(self, o: object, /) -> bool:
        ...

    def __contains__(self, o: Any, /) -> bool:
        """Check if a key is in the ParamBounds instance."""
        if isinstance(o, str):
            return bool(super().__contains__(o))
        else:
            try:
                self[o]
            except KeyError:
                return False
            else:
                return True

    # =========================================================================
    # Flat

    def flatitems(
        self,
    ) -> Iterable[tuple[tuple[str | T] | tuple[str | T, str], Object]]:
        """Flattened items."""
        for name, bounds in self.items():
            if isinstance(bounds, self._Object):  # FIXME!
                yield (name,), bounds
            elif isinstance(bounds, Mapping):
                for subname, subbounds in bounds.items():
                    yield (name, subname), subbounds
            else:
                msg = f"Unexpected type: {type(bounds)}"
                raise TypeError(msg)

    def flatkeys(self) -> tuple[tuple[str | T] | tuple[str | T, str], ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatitems())

    def flatvalues(self) -> tuple[Object, ...]:
        """Flattened values."""
        return tuple(v for _, v in self.flatitems())

    # =========================================================================
    # Misc

    def validate(self, names: ParamNames, *, error: bool = False) -> bool | None:
        """Check that the parameter bounds are consistent with the model."""
        if self.flatkeys() != names.flats:
            if not error:
                return False

            # TODO: more informative error.
            msg = "param_bounds keys do not match param_names"
            raise ValueError(msg)

        return True
