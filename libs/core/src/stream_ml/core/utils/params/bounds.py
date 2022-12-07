"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import ItemsView, Iterable, KeysView, Mapping, ValuesView
from typing import TYPE_CHECKING, Any, TypeAlias, cast, overload

# LOCAL
from stream_ml.core.utils.hashdict import FrozenDict

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.utils.params.names import ParamNames

__all__: list[str] = []

SubParamBounds: TypeAlias = FrozenDict[str, tuple[float, float]]

inf = float("inf")


class ParamBounds(FrozenDict[str, tuple[float, float] | SubParamBounds]):
    """A frozen (hashable) dictionary of parameters."""

    def __init__(self, m: Any = (), /, **kwargs: Any) -> None:
        # Initialize, with processing, then validate
        super().__init__(
            **{
                k: FrozenDict(v) if isinstance(v, Mapping) else v
                for k, v in dict(m, **kwargs).items()
            },
        )

        # Validate structure
        for k, v in self.items():
            if not isinstance(k, str):
                raise TypeError(f"Invalid key type: {type(k)}")
            if isinstance(v, tuple):
                if len(v) != 2 or not all(isinstance(e, (int, float)) for e in v):
                    raise ValueError(f"Invalid value: {v}")
            elif isinstance(v, FrozenDict):
                for k2, v2 in v.items():
                    if not isinstance(k2, str):
                        raise TypeError(f"Invalid key type: {type(k2)}")
                    elif (
                        not isinstance(v2, tuple)
                        or len(v2) != 2
                        or not all(isinstance(e, (int, float)) for e in v2)
                    ):
                        raise ValueError(f"Invalid value: {k2} = {v2}")
            else:
                raise TypeError(f"Invalid element type: {type(v)}")

    @classmethod
    def from_names(cls, names: ParamNames) -> ParamBounds:
        """Create a new ParamBounds instance."""
        pbs = cls()
        for pn in names:
            if isinstance(pn, str):  # e.g. "mixparam"
                pbs._mapping[pn] = (-inf, inf)
            else:  # e.g. ("phi2", ("mu", "sigma"))
                pbs._mapping[pn[0]] = FrozenDict[str, tuple[float, float]](
                    {k: (-inf, inf) for k in pn[1]}
                )

        return cls(pbs)

    # =========================================================================

    @overload
    def __getitem__(self, key: str) -> tuple[float, float] | SubParamBounds:
        ...

    @overload
    def __getitem__(self, key: tuple[str]) -> tuple[float, float]:
        ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> tuple[float, float]:
        ...

    def __getitem__(
        self, key: str | tuple[str] | tuple[str, str]
    ) -> tuple[float, float] | SubParamBounds:
        if isinstance(key, str):
            return super().__getitem__(key)
        elif len(key) == 1:
            v = super().__getitem__(key[0])
            if not isinstance(v, tuple):
                raise KeyError(key)
            return v
        else:
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            v = super().__getitem__(key[0])
            if not isinstance(v, FrozenDict):
                raise KeyError(key)
            return v[key[1]]

    def keys(self) -> KeysView[str]:
        """Parameter bounds keys."""
        return self._mapping.keys()

    def values(self) -> ValuesView[tuple[float, float] | SubParamBounds]:
        """Parameter bounds values."""
        return self._mapping.values()

    def items(self) -> ItemsView[str, tuple[float, float] | SubParamBounds]:
        """Parameter bounds items."""
        return self._mapping.items()

    # =========================================================================

    def __or__(self, other: Any) -> ParamBounds:
        """Combine two ParamBounds instances."""
        if not isinstance(other, ParamBounds):
            raise NotImplementedError

        pbs = type(self)(**self._mapping)

        for k, v in other.items():
            if k not in pbs or isinstance(v, tuple):
                pbs._mapping[k] = v
                continue
            elif isinstance((sv := pbs._mapping[k]), tuple):
                raise ValueError(f"mixing tuple and FrozenDict is not allowed: {k}")
            else:
                pbs._mapping[k] = sv | v

        return pbs

    @overload
    def __contains__(self, o: str, /) -> bool:
        ...

    @overload
    def __contains__(self, o: tuple[str] | tuple[str, str], /) -> bool:
        ...

    @overload
    def __contains__(self, o: object, /) -> bool:
        ...

    def __contains__(self, o: Any, /) -> bool:
        return super().__contains__(o)

    # =========================================================================

    def flatitems(
        self,
    ) -> Iterable[tuple[tuple[str] | tuple[str, str], tuple[float, float]]]:
        """Flattened items."""
        for name, bounds in self.items():
            if isinstance(bounds, tuple):
                yield (name,), bounds
            else:
                for subname, subbounds in bounds.items():
                    yield (name, subname), subbounds

    def flatkeys(self) -> tuple[tuple[str] | tuple[str, str], ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatitems())

    def flatvalues(self) -> tuple[tuple[float, float], ...]:
        """Flattened values."""
        return tuple(v for _, v in self.flatitems())


class ParamBoundsField:
    """Dataclass descriptor for parameter bounds."""

    def __init__(
        self,
        default: ParamBounds
        | Mapping[str, tuple[float, float] | Mapping[str, tuple[float, float]]]
        | None = None,
    ) -> None:
        self._default: ParamBounds | None
        self._default = ParamBounds(default) if default is not None else None

    def __set_name__(self, _: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, _: type | None) -> ParamBounds:
        if obj is not None:
            val: ParamBounds = getattr(obj, self._name)
            return val

        default = self._default
        if default is not None:
            return default
        else:
            raise AttributeError(f"no default value for {self._name}")

    def __set__(self, obj: object, value: ParamBounds) -> None:
        dv = ParamBounds(self._default or {})  # Default value as a dict.
        object.__setattr__(obj, self._name, dv | value)
