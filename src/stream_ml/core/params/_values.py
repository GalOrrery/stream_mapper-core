"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from collections.abc import Iterable, Mapping
import itertools
from typing import TYPE_CHECKING, TypeVar, cast, overload

from stream_ml.core.setup_package import PACK_PARAM_JOIN
from stream_ml.core.utils.frozen_dict import FrozenDict

if TYPE_CHECKING:
    from stream_ml.core.typing import ParamNameAllOpts, ParamNameTupleOpts


V = TypeVar("V")
LEN_NAME_TUPLE: int = 2


class Params(FrozenDict[str, V | FrozenDict[str, V]]):
    """Parameter dictionary."""

    def __init__(
        self,
        m: Mapping[str, V | Mapping[str, V]] = {},
        /,
        **kwargs: V | Mapping[str, V],
    ) -> None:
        # Shortcut if `m` is Params and there's no kwargs
        if isinstance(m, Params) and not kwargs:
            super().__init__(m._dict, __unsafe_skip_copy__=True)
            return

        # Freeze sub-dicts
        d: dict[str, V | FrozenDict[str, V]] = {
            k: (v if not isinstance(v, Mapping) else FrozenDict[str, V](v))
            for k, v in itertools.chain(m.items(), kwargs.items())
        }
        super().__init__(d, __unsafe_skip_copy__=True)

    # -----------------------------------------------------

    @overload
    def __getitem__(self, key: str) -> V | FrozenDict[str, V]:
        ...

    @overload
    def __getitem__(self, key: tuple[str]) -> V:
        ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> V:
        ...

    def __getitem__(self, key: ParamNameAllOpts) -> V | FrozenDict[str, V]:
        if isinstance(key, str):
            value = self._dict[key]
        elif len(key) == 1:
            value = self._dict[key[0]]
        elif len(key) == LEN_NAME_TUPLE:
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            cm = self._dict[key[0]]
            if not isinstance(cm, Mapping):
                raise KeyError(str(key))
            value = cm[key[1]]
        else:
            raise KeyError(str(key))
        return value

    def unfreeze(self) -> dict[str, V | dict[str, V]]:  # type: ignore[override]
        """Unfreeze the parameters."""
        return unfreeze_params(self)

    # =========================================================================
    # Flats
    # Tuple keys are used to access the parameters.

    # TODO: cache, speed up, and ItemsView
    def flatsitems(self) -> tuple[tuple[ParamNameTupleOpts, V], ...]:
        """Flattened items."""
        return tuple(_flats_iter(self))

    # TODO: cache
    def flatskeys(self) -> tuple[ParamNameTupleOpts, ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatsitems())

    # TODO: cache
    def flatsvalues(self) -> tuple[V, ...]:
        """Flattened values."""
        return tuple(v for _, v in self.flatsitems())

    # =========================================================================
    # Flat

    def flatitems(self) -> tuple[tuple[str, V], ...]:
        """Flat items."""
        return tuple((PACK_PARAM_JOIN.join(k), v) for k, v in _flats_iter(self))

    # TODO: cache
    def flatkeys(self) -> tuple[str, ...]:
        """Flat keys."""
        return tuple(k for k, _ in self.flatitems())

    # TODO: cache
    def flatvalues(self) -> tuple[V, ...]:
        """Flat values."""
        return tuple(v for _, v in self.flatitems())

    # =========================================================================

    def get_prefixed(self, prefix: str) -> Params[V]:
        """Get the keys starting with the prefix, stripped of that prefix."""
        prefix = prefix + "." if not prefix.endswith(".") else prefix
        lp = len(prefix)
        return type(self)({k[lp:]: v for k, v in self.items() if k.startswith(prefix)})

    def add_prefix(self, prefix: str, /) -> Params[V]:
        """Add the prefix to the keys."""
        return add_prefix(self, prefix)


def _flats_iter(
    params: Params[V], /
) -> Iterable[tuple[tuple[str], V] | tuple[tuple[str, str], V]]:
    for k, v in params.items():
        if not isinstance(v, Mapping):
            yield (k,), v
        else:
            for kk, vv in v.items():
                yield (k, kk), vv


#####################################################################


def freeze_params(m: Mapping[str, V | Mapping[str, V]], /) -> Params[V]:
    """Freeze a mapping of parameters."""
    return Params(m)


def unfreeze_params(
    pars: Params[V],
    /,
) -> dict[str, V | dict[str, V]]:
    """Unfreeze a mapping of parameters."""
    return {k: (v if not isinstance(v, Mapping) else dict(v)) for k, v in pars.items()}


# -----------------------------------------------------


@overload
def set_param(
    m: dict[str, V | dict[str, V]],
    /,
    key: ParamNameAllOpts,
    value: V | dict[str, V],
) -> dict[str, V | dict[str, V]]:
    ...


@overload
def set_param(
    m: Params[V],
    /,
    key: ParamNameAllOpts,
    value: V | dict[str, V],
) -> Params[V]:
    ...


def set_param(
    m: dict[str, V | dict[str, V]] | Params[V],
    /,
    key: ParamNameAllOpts,
    value: V | dict[str, V],
) -> dict[str, V | dict[str, V]] | Params[V]:
    """Set a parameter on a Params or Params-like dictionary.

    Parameters
    ----------
    m : MutableMapping[str, V | MutableMapping[str, V]], positional-only
        The dictionary to set the parameter on.
    key : ParamNameAllOpts
        The key to set.
    value : V | dict[str, V]
        The value to set.

    Returns
    -------
    Mapping[str, V | Mapping[str, V]]
    """
    return (
        _set_param_params(m, key, value)
        if isinstance(m, Params)
        else _set_param_dict(m, key, value)
    )


def _set_param_dict(
    m: dict[str, V | dict[str, V]],
    /,
    key: ParamNameAllOpts,
    value: V | dict[str, V],
) -> dict[str, V | dict[str, V]]:
    if isinstance(key, str):
        m[key] = value
    elif len(key) == 1:
        m[key[0]] = value
    else:
        key = cast("tuple[str, str]", key)  # TODO: remove cast
        if key[0] not in m:
            m[key[0]] = {}
        if not isinstance((cm := m[key[0]]), dict):
            raise KeyError(str(key))
        cm[key[1]] = value  # type: ignore[assignment]

    return m


def _set_param_params(
    m: Params[V],
    /,
    key: ParamNameAllOpts,
    value: V | dict[str, V],
) -> Params[V]:
    if isinstance(key, str) or len(key) == 1:
        # We can shortcut copying sub-dicts
        pum = dict(m._dict.items())
        k = key if isinstance(key, str) else key[0]
        pum[k] = FrozenDict(value) if isinstance(value, dict) else value
        # Note this copies the dict one more time. It would be nice to avoid this.
        return type(m)(pum)
    else:
        # Note this copies the dict one more time. It would be nice to avoid this.
        return type(m)(set_param(m.unfreeze(), key, value))


# -----------------------------------------------------


M = TypeVar("M", dict[str, V], Params[V])  # type: ignore[valid-type]


def add_prefix(m: M, /, prefix: str) -> M:
    """Add the prefix to the keys.

    Parameters
    ----------
    m : Mapping, positional-only
        The mapping to add the prefix to. Keys must be strings.
    prefix : str
        The prefix to add.

    Returns
    -------
    Mapping
        The mapping with the prefix added to the keys.
        Same type as the input mapping.
    """
    return m.__class__({f"{prefix}{k}": v for k, v in m.items()})
