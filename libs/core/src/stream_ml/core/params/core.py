"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    MutableMapping,
    ValuesView,
)
from typing import Any, TypeVar, cast, overload

__all__: list[str] = []


V = TypeVar("V")


class Params(Mapping[str, V | Mapping[str, V]]):
    """Parameter dictionary."""

    __slots__ = ("_dict",)

    def __init__(self, m: Any = (), /, **kwargs: V | Mapping[str, V]) -> None:
        self._dict: MutableMapping[str, V | Mapping[str, V]] = dict(m, **kwargs)

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
            value = self._dict[key]
        elif len(key) == 1:
            value = self._dict[key[0]]
        elif len(key) == 2:
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            cm = self._dict[key[0]]
            if not isinstance(cm, Mapping):
                raise KeyError(str(key))
            value = cm[key[1]]
        else:
            raise KeyError(str(key))
        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys."""
        return iter(self._dict)

    def __len__(self) -> int:
        """Length."""
        return len(self._dict)

    def __repr__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._dict!r})"

    def keys(self) -> KeysView[str]:
        """Keys."""
        return self._dict.keys()

    def values(self) -> ValuesView[V | Mapping[str, V]]:
        """Values."""
        return self._dict.values()

    def items(self) -> ItemsView[str, V | Mapping[str, V]]:
        """Items."""
        return self._dict.items()

    # =========================================================================
    # Flat

    def flatitems(self) -> Iterable[tuple[str, V]]:
        """Flat items."""
        for k, v in self._dict.items():
            if not isinstance(v, Mapping):
                yield k, v
            else:
                for k2, v2 in v.items():
                    yield f"{k}_{k2}", v2

    def flatkeys(self) -> Iterable[str]:
        """Flat keys."""
        for k, _ in self.flatitems():
            yield k

    def flatvalues(self) -> Iterable[V]:
        """Flat values."""
        for _, v in self.flatitems():
            yield v

    # =========================================================================

    def get_prefixed(self, prefix: str) -> Params[V]:
        """Get the keys starting with the prefix, stripped of that prefix."""
        prefix = prefix + "." if not prefix.endswith(".") else prefix
        lp = len(prefix)
        return Params(
            {k[lp:]: v for k, v in self._dict.items() if k.startswith(prefix)}
        )

    def add_prefix(self, prefix: str, /) -> Params[V]:
        """Add the prefix to the keys."""
        return type(self)({f"{prefix}{k}": v for k, v in self.items()})


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
            self._dict[key] = value
        elif len(key) == 1:
            self._dict[key[0]] = value
        else:
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            if key[0] not in self._dict:
                self._dict[key[0]] = {}
            if not isinstance((cm := self._dict[key[0]]), MutableMapping):
                raise KeyError(str(key))
            cm[key[1]] = value

    def __delitem__(self, key: str | tuple[str] | tuple[str, str]) -> None:
        if isinstance(key, str):
            del self._dict[key]
        elif len(key) == 1:
            del self._dict[key[0]]
        else:
            key = cast("tuple[str, str]", key)
            cm = self._dict[key[0]]
            if not isinstance(cm, MutableMapping):
                raise KeyError(str(key))
            del cm[key[1]]
