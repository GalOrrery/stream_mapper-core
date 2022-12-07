"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Iterable
from typing import Any, TypeVar

__all__: list[str] = []


Self = TypeVar("Self", bound="ParamNames")


class ParamNames(tuple[str | tuple[str, tuple[str, ...]], ...]):
    """Parameter names."""

    def __self__(self, iterable: Any = (), /) -> None:
        """Create a new ParamNames instance."""
        super().__init__()

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
