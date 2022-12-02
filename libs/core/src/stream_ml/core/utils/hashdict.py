"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Generic, TypeVar

__all__: list[str] = []

K = TypeVar("K")
V = TypeVar("V")


class HashableMap(Mapping[K, V]):
    """A frozen (hashable) dictionary."""

    __slots__ = ("_mapping",)

    def __init__(self, m: Any = (), /, **kwargs: Any) -> None:
        # Please do not mutate this dictionary.
        self._mapping: dict[K, V] = dict(m, **kwargs)
        # Make sure that the dictionary is hashable.
        hash(self)
        return None

    def __iter__(self) -> Iterator[K]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(self, key: K) -> V:
        return self._mapping[key]

    def __hash__(self) -> int:
        return hash(tuple(self._mapping.items()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._mapping!r})"

    def __or__(self, other: Mapping[K, V]) -> HashableMap[K, V]:
        if not isinstance(other, HashableMap):
            raise NotImplementedError
        return HashableMap(self._mapping | dict(other))


class HashableMapField(Generic[K, V]):
    """Dataclass descriptor for a frozen map."""

    def __init__(
        self, default: Mapping[K, V] | Sequence[tuple[K, V]] | None = None
    ) -> None:
        self._default: HashableMap[K, V] | None
        self._default = HashableMap(default) if default is not None else None

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, obj_cls: type | None) -> HashableMap[K, V]:
        if obj is not None:
            val: HashableMap[K, V] = getattr(obj, self._name)
            return val

        default = self._default
        if default is None:
            raise AttributeError(f"no default value for {self._name}")
        return default

    def __set__(self, obj: object, value: Mapping[K, V]) -> None:
        dv = HashableMap[K, V](self._default or {})  # Default value as a dict.
        object.__setattr__(obj, self._name, dv | HashableMap(value))
