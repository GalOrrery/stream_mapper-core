"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Any, TypeVar, cast, overload

# LOCAL
from stream_ml.core.utils.hashdict import HashableMap

__all__: list[str] = []


V = TypeVar("V")


class Params(Mapping[str, V | Mapping[str, V]]):
    """Parameter dictionary."""

    __slots__ = ("_mapping",)

    def __init__(self, m: Any = (), /, **kwargs: V | Mapping[str, V]) -> None:
        self._mapping: MutableMapping[str, V | Mapping[str, V]] = dict(m, **kwargs)

        # TODO: Validation

    @overload
    def __getitem__(self, key: str) -> V | Mapping[str, V]:
        ...

    @overload
    def __getitem__(self, key: tuple[str]) -> V:
        ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> V:
        ...

    def __getitem__(
        self, key: str | tuple[str] | tuple[str, str]
    ) -> V | Mapping[str, V]:
        if isinstance(key, str):
            return self._mapping[key]
        elif len(key) == 1:
            return self._mapping[key[0]]
        elif len(key) == 2:
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            cm = self._mapping[key[0]]
            if not isinstance(cm, Mapping):
                raise KeyError(str(key))
            return cm[key[1]]
        raise KeyError(str(key))

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys."""
        return iter(self._mapping)

    def __len__(self) -> int:
        """Length."""
        return len(self._mapping)

    def get_prefixed(self, prefix: str) -> Params[V]:
        """Get the keys starting with the prefix, stripped of that prefix."""
        prefix = prefix + "_" if not prefix.endswith("_") else prefix
        lp = len(prefix)
        return Params(
            {k[lp:]: v for k, v in self._mapping.items() if k.startswith(prefix)}
        )

    def add_prefix(self, prefix: str, inplace: bool = False) -> Params[V]:
        """Add the prefix to the keys."""
        if inplace:
            for k in tuple(self._mapping.keys()):
                self._mapping[prefix + k] = self._mapping.pop(k)
            return self

        return Params({f"{prefix}{k}": v for k, v in self._mapping.items()})


class MutableParams(Params[V], MutableMapping[str, V | MutableMapping[str, V]]):
    """Mutable Params."""

    @overload
    def __setitem__(self, key: str, value: V | MutableMapping[str, V]) -> None:
        ...

    @overload
    def __setitem__(self, key: tuple[str], value: V | MutableMapping[str, V]) -> None:
        ...

    @overload
    def __setitem__(self, key: tuple[str, str], value: V) -> None:
        ...

    def __setitem__(
        self, key: str | tuple[str] | tuple[str, str], value: V | MutableMapping[str, V]
    ) -> None:
        if isinstance(key, str):
            self._mapping[key] = value
        elif len(key) == 1:
            self._mapping[key[0]] = value
        else:
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            if key[0] not in self._mapping:
                self._mapping[key[0]] = {}
            if not isinstance((cm := self._mapping[key[0]]), MutableMapping):
                raise KeyError(str(key))
            cm[key[1]] = value

    def __delitem__(self, key: str | tuple[str] | tuple[str, str]) -> None:
        if isinstance(key, str):
            del self._mapping[key]
        elif len(key) == 1:
            del self._mapping[key[0]]
        else:
            key = cast("tuple[str, str]", key)
            cm = self._mapping[key[0]]
            if not isinstance(cm, MutableMapping):
                raise KeyError(str(key))
            del cm[key[1]]


# =============================================================================


Self = TypeVar("Self", bound="ParamNames")


class ParamNames(tuple[str | tuple[str, tuple[str, ...]], ...]):
    """A frozen (hashable) dictionary."""

    def __new__(cls: type[Self], iterable: Any = (), /) -> Self:
        """Create a new ParamNames instance."""
        self: Self = super().__new__(cls, iterable)

        # Validate structure
        for elt in self:
            if isinstance(elt, str):
                continue
            elif not isinstance(elt, tuple):
                raise TypeError(f"Invalid element type: {type(elt)}")
            elif len(elt) != 2:
                raise ValueError(f"Invalid element length: {len(elt)}")
            elif not isinstance(elt[0], str):
                raise TypeError(f"Invalid element type: {type(elt[0])}")
            elif not isinstance(elt[1], tuple):
                raise TypeError(f"Invalid element type: {type(elt[1])}")
            elif not all(isinstance(e, str) for e in elt[1]):
                raise TypeError(f"Invalid element type: {type(elt[1])}")

        return self

    @property
    def flat(self) -> tuple[str, ...]:
        """Flattened parameter names."""
        names: list[str] = []

        for pn in self:
            if isinstance(pn, str):
                names.append(pn)
            else:
                names.extend(f"{pn[0]}_{k}" for k in pn[1])

        return tuple(names)

    @property
    def flats(self) -> tuple[tuple[str] | tuple[str, str], ...]:
        """Flattened parameter names as tuples."""
        names: list[tuple[str] | tuple[str, str]] = []

        for pn in self:
            if isinstance(pn, str):
                names.append((pn,))
            else:
                names.extend((pn[0], k) for k in pn[1])

        return tuple(names)


class ParamNamesField:
    """Dataclass descriptor for a frozen map."""

    def __init__(
        self,
        default: ParamNames | Iterable[str | tuple[str, tuple[str, ...]]] | None = None,
    ) -> None:
        self._default: ParamNames | None
        self._default = ParamNames(default) if default is not None else None

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, obj_cls: type | None) -> ParamNames:
        if obj is not None:
            val: ParamNames = getattr(obj, self._name)
            return val

        default = self._default
        if default is not None:
            return default
        else:
            raise AttributeError(f"no default value for {self._name}")

    def __set__(self, obj: object, value: ParamNames) -> None:
        object.__setattr__(obj, self._name, ParamNames(value))


# =============================================================================


class ParamBounds(
    HashableMap[str, tuple[float, float] | HashableMap[str, tuple[float, float]]]
):
    """A frozen (hashable) dictionary of parameters."""

    def __init__(self, m: Any = (), /, **kwargs: Any) -> None:
        # Initialize, with processing, then validate
        super().__init__(
            **{
                k: HashableMap(v) if isinstance(v, Mapping) else v
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
            elif isinstance(v, HashableMap):
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
                pbs._mapping[pn] = (-float("inf"), float("inf"))
            else:  # e.g. ("phi2", ("mu", "sigma"))
                pbs._mapping[pn[0]] = HashableMap[str, tuple[float, float]](
                    {k: (-float("inf"), float("inf")) for k in pn[1]}
                )

        return cls(pbs)

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
                raise ValueError(f"mixing tuple and HashableMap is not allowed: {k}")
            else:
                pbs._mapping[k] = sv | v

        return pbs

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

    @overload
    def __getitem__(
        self, key: str
    ) -> tuple[float, float] | HashableMap[str, tuple[float, float]]:
        ...

    @overload
    def __getitem__(self, key: tuple[str]) -> tuple[float, float]:
        ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> tuple[float, float]:
        ...

    def __getitem__(
        self, key: str | tuple[str] | tuple[str, str]
    ) -> tuple[float, float] | HashableMap[str, tuple[float, float]]:
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
            if not isinstance(v, HashableMap):
                raise KeyError(key)
            return v[key[1]]


class ParamBoundsField:
    """Dataclass descriptor for a frozen map."""

    def __init__(
        self,
        default: ParamBounds
        | Mapping[str, tuple[float, float] | Mapping[str, tuple[float, float]]]
        | None = None,
    ) -> None:
        self._default: ParamBounds | None
        self._default = ParamBounds(default) if default is not None else None

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, obj_cls: type | None) -> ParamBounds:
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
