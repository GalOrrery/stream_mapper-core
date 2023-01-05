"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from typing import Generic, Literal, Protocol, TypeVar

# LOCAL
from stream_ml.core.utils.sentinel import MISSING, Sentinel

__all__: list[str] = []

K = TypeVar("K")
V = TypeVar("V")
_VT_co = TypeVar("_VT_co", covariant=True)


class SupportsKeysAndGetItem(Protocol[K, _VT_co]):
    """Protocol for ``keys()`` and ``__getitem__`` support.

    This is a subset of the ``Mapping`` protocol and the minimum requirement
    for input to ``FrozenDict``.
    """

    def keys(self) -> Iterable[K]:
        """Return keys."""
        ...

    def __getitem__(self, __key: K) -> _VT_co:
        """Get item."""
        ...


class FrozenDict(Mapping[K, V]):
    """A frozen (hashable) dictionary."""

    __slots__ = ("_dict", "_hash")

    def __init__(
        self,
        m: SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]] = (),
        /,
        **kwargs: V,
    ) -> None:
        # Please do not mutate this dictionary.
        self._dict: dict[K, V] = dict(m, **kwargs)
        # Make sure that the dictionary is hashable.
        hash(self)
        return

    def __iter__(self) -> Iterator[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __getitem__(self, key: K) -> V:
        return self._dict[key]

    def __hash__(self) -> int:
        return hash(tuple(self._dict.items()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._dict!r})"

    def __or__(self, other: Mapping[K, V]) -> FrozenDict[K, V]:
        if not isinstance(other, FrozenDict):
            raise NotImplementedError
        return FrozenDict(self._dict | dict(other))

    def keys(self) -> KeysView[K]:
        """Return keys view."""
        return self._dict.keys()

    def values(self) -> ValuesView[V]:
        """Return values view."""
        return self._dict.values()

    def items(self) -> ItemsView[K, V]:
        """Return items view."""
        return self._dict.items()


class FrozenDictField(Generic[K, V]):
    """Dataclass descriptor for a frozen map."""

    def __init__(
        self,
        default: Mapping[K, V]
        | Sequence[tuple[K, V]]
        | Literal[Sentinel.MISSING] = MISSING,
    ) -> None:
        self._default: FrozenDict[K, V] | Literal[Sentinel.MISSING]
        self._default = FrozenDict(default) if default is not MISSING else MISSING

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, obj_cls: type | None) -> FrozenDict[K, V]:
        if obj is not None:
            val: FrozenDict[K, V] = getattr(obj, self._name)
            return val

        default = self._default
        if default is MISSING:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)
        return default

    def __set__(self, obj: object, value: Mapping[K, V]) -> None:
        # Default value as a dict.
        dv = FrozenDict[K, V](self._default if self._default is not MISSING else {})
        # Set the value. This is only called once by the dataclass.
        object.__setattr__(obj, self._name, dv | FrozenDict(value))