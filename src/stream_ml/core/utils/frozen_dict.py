"""Frozen dictionary.

Modified from :mod:`~flax`, with the following license:
::

    Copyright 2022 The Flax Authors.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

Main changes:

- static typing
- ``__or__`` method
- FrozenItemsView
- FrozenDictField
- default value for ``add_or_replace`` in ``copy``
- removed jax stuff
"""

from __future__ import annotations

__all__ = ["FrozenDict", "freeze", "unfreeze"]

from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
import textwrap
from typing import Any, Generic, Protocol, TypeVar

from stream_ml.core.utils.sentinel import MISSING, MissingT

###############################################################################


K = TypeVar("K")
V = TypeVar("V")
Self = TypeVar("Self", bound="FrozenDict[K, V]")  # type: ignore[valid-type]
_VT_co = TypeVar("_VT_co", covariant=True)


###############################################################################


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


# ===================================================================


class FrozenKeysView(KeysView[K]):
    """A wrapper for a more useful repr of the keys in a frozen dict."""

    def __iter__(self) -> Iterator[K]:
        return super().__iter__()

    def __repr__(self) -> str:
        return f"frozen_dict_keys({list(self)})"


class FrozenValuesView(ValuesView[V]):
    """A wrapper for a more useful repr of the values in a frozen dict."""

    def __iter__(self) -> Iterator[V]:
        return super().__iter__()

    def __repr__(self) -> str:
        return f"frozen_dict_values({list(self)})"


class FrozenItemsView(ItemsView[K, V]):
    """A wrapper for a more useful repr of the items in a frozen dict."""

    def __iter__(self) -> Iterator[tuple[K, V]]:
        return super().__iter__()

    def __repr__(self) -> str:
        return f"frozen_dict_items({list(self)})"


# ===================================================================


class FrozenDict(Mapping[K, V]):
    """A frozen (hashable) dictionary.

    Parameters
    ----------
    m: SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]], optional positional-only
        Mapping argument. See ``dict`` for details.
    __unsafe_skip_copy__: bool, optional keyword-only
        If ``True``, the input mapping is used directly. This is unsafe because
        the input mapping may be mutated after the ``FrozenDict`` is created.
        This is used internally to avoid copying the input mapping when it is
        already a ``FrozenDict``. Default is ``False``. This argument is
        private and should not be used.
    **kwargs: V, optional keyword-only
        Additional keyword arguments. See ``dict`` for details.
    """

    __slots__ = ("_dict", "_hash")

    def __init__(
        self,
        m: SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]] = (),
        /,
        *,
        __unsafe_skip_copy__: bool = False,
        **kwargs: V,
    ) -> None:
        xs: dict[K, V] = dict(m, **kwargs)
        self._dict = xs if __unsafe_skip_copy__ else _prepare_freeze(xs)
        self._hash: int | None = None

    def __iter__(self) -> Iterator[K]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __getitem__(self, key: K) -> V:
        # if isinstance(v, dict):  # TODO: in flax but hard to type
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def __hash__(self) -> int:
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._dict!r})"

    def __or__(self: Self, other: Any | FrozenDict[K, V]) -> Self:
        if not isinstance(other, type(self)):
            return NotImplemented
        return type(self)(self._dict | other._dict)

    def __reduce__(self) -> tuple[type, tuple[dict[K, V]]]:
        return type(self), (self.unfreeze(),)

    # ===================================================================

    def keys(self) -> KeysView[K]:
        """Return keys view."""
        return FrozenKeysView(self)

    def values(self) -> ValuesView[V]:
        """Return values view."""
        return FrozenValuesView(self)

    def items(self) -> ItemsView[K, V]:
        """Return items view."""
        return FrozenItemsView(self)

    def copy(self, add_or_replace: Mapping[K, V] | None = None) -> FrozenDict[K, V]:
        """Create a new FrozenDict with additional or replaced entries."""
        return type(self)({**self, **(add_or_replace or {})})

    def pop(self, key: K) -> tuple[FrozenDict[K, V], V]:
        """Create a new FrozenDict where one entry is removed.

        Parameters
        ----------
        key: K
            the key to remove from the dict

        Returns
        -------
        new_self: FrozenDict[K, V]
            A new FrozenDict with the removed value.
        value: V
            The removed value.

        Examples
        --------
        >>> d = FrozenDict({"a": 1, "b": 2})
        >>> d.pop("a")
        FrozenDict({'b': 2}), 1
        """
        value = self[key]
        new_dict = dict(self._dict)
        new_dict.pop(key)
        new_self = type(self)(new_dict)
        return new_self, value

    def unfreeze(self) -> dict[K, V]:
        """Unfreeze this FrozenDict.

        Return:
        ------
        dict[K, V]
            An unfrozen version of this FrozenDict instance.
        """
        return unfreeze(self, deep=True)

    # ===================================================================

    def __str__(self) -> str:
        s = f"{self.__class__.__name__}(" + "{\n"  # noqa: ISC003
        s += "\n".join(
            textwrap.indent(f"{k}: {v!s}", prefix="\t") for k, v in self.items()
        )
        s += "\n})"
        return s


# ===================================================================


def _recursive_prepare_freeze(xs: dict[K, V], /) -> dict[K, V]:
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    # recursively copy dictionary to avoid ref sharing
    return {
        k: (
            _recursive_prepare_freeze(v)  # type: ignore[misc]
            if isinstance(v, dict)
            else v
        )
        for k, v in xs.items()
    }


def _prepare_freeze(xs: dict[K, V] | FrozenDict[K, V]) -> dict[K, V]:
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    if isinstance(xs, FrozenDict):
        return xs._dict
    return _recursive_prepare_freeze(xs)


def freeze(m: dict[K, V], /) -> FrozenDict[K, V]:
    """Freeze a nested dict.

    Makes a nested `dict` immutable by transforming it into `FrozenDict`.

    Parameters
    ----------
    m: dict[K, V]
        Dictionary to freeze.

    Returns
    -------
    FrozenDict
        The frozen dictionary.
    """
    return FrozenDict(m)


# TODO! actually do the recursive type
def _recursive_unfreeze(x: Mapping[K, V], /, *, deep: bool) -> dict[K, V]:
    ys: dict[K, V] = {}
    for k, v in x.items():
        if isinstance(v, dict) or (deep and isinstance(v, FrozenDict)):
            ys[k] = _recursive_unfreeze(v, deep=deep)  # type: ignore[assignment]
        else:
            ys[k] = v
    return ys


def unfreeze(x: FrozenDict[K, V], *, deep: bool = False) -> dict[K, V]:
    """Unfreeze a FrozenDict.

    Makes a mutable copy of a `FrozenDict` mutable by transforming
    it into (nested) dict.

    Parameters
    ----------
    x: FrozenDict
        Frozen dictionary to unfreeze.

    deep : bool, optional keyword-only
        Whether to unfreeze FrozenDict recursively. Defaults to False.

    Returns
    -------
    dict
        The unfrozen dictionary.
    """
    return _recursive_unfreeze(x, deep=deep)


###############################################################################


class FrozenDictField(Generic[K, V]):
    """Dataclass descriptor for a frozen map."""

    def __init__(
        self,
        default: dict[K, V]
        | FrozenDict[K, V]
        | Sequence[tuple[K, V]]
        | MissingT = MISSING,
    ) -> None:
        self._default: FrozenDict[K, V] | MissingT
        self._default = FrozenDict(default) if default is not MISSING else MISSING

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, obj_cls: type | None) -> FrozenDict[K, V]:
        # Get field, which is a FrozenDict.
        if obj is not None:
            val: FrozenDict[K, V] = getattr(obj, self._name)
            return val

        # Get default value, when setting.
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
