"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import ItemsView, Iterable, KeysView, Mapping, ValuesView
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Generic, cast, overload

# LOCAL
from stream_ml.core._typing import Array
from stream_ml.core.prior.bounds import PriorBounds
from stream_ml.core.utils.hashdict import FrozenDict

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.params.names import ParamNames

__all__: list[str] = []

#####################################################################
# PARAMETERS

inf = float("inf")


#####################################################################


class ParamBounds(
    FrozenDict[str, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]]
):
    """A frozen (hashable) dictionary of parameters."""

    def __init__(self, m: Any = (), /, **kwargs: Any) -> None:
        # Initialize, with processing, then validate
        # TODO: not cast to dict if already a ParamBounds
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
            if isinstance(v, PriorBounds):
                pass
            elif isinstance(v, FrozenDict):
                for k2, v2 in v.items():
                    if not isinstance(k2, str):
                        raise TypeError(f"Invalid key type: {type(k2)}")
                    elif not isinstance(v2, PriorBounds) or not all(
                        isinstance(e, (int, float)) for e in v2
                    ):
                        raise ValueError(f"Invalid value: {k2} = {v2}")
            else:
                raise TypeError(f"Invalid element type: {type(v)}")

    @classmethod
    def from_names(
        cls, names: ParamNames, default: PriorBounds[Array]
    ) -> ParamBounds[Array]:
        """Create a new ParamBounds instance.

        Parameters
        ----------
        names : ParamNames, positional-only
            The parameter names.
        default : PriorBounds
            The default prior bounds.

        Returns
        -------
        ParamBounds
        """
        m: dict[str, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]] = {}
        for pn in names:
            if isinstance(pn, str):  # e.g. "mixparam"
                m[pn] = replace(default, param_name=(pn,))
            else:  # e.g. ("phi2", ("mu", "sigma"))
                m[pn[0]] = FrozenDict(
                    {k: replace(default, param_name=(pn[0], k)) for k in pn[1]}
                )

        return cls(m)

    # =========================================================================
    # Mapping

    @overload
    def __getitem__(
        self, key: str
    ) -> PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]:
        ...

    @overload
    def __getitem__(self, key: tuple[str]) -> PriorBounds[Array]:  # Flat key
        ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> PriorBounds[Array]:  # Flat key
        ...

    def __getitem__(
        self, key: str | tuple[str] | tuple[str, str]
    ) -> PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]:
        if isinstance(key, str):  # e.g. "mixparam"
            return super().__getitem__(key)
        elif len(key) == 1:  # e.g. ("mixparam",)
            v = super().__getitem__(key[0])
            if not isinstance(v, PriorBounds):
                raise KeyError(key)
            return v
        else:  # e.g. ("phi2", "mu")
            key = cast("tuple[str, str]", key)  # TODO: remove cast
            v = super().__getitem__(key[0])
            if not isinstance(v, FrozenDict):
                raise KeyError(key)
            return v[key[1]]

    def keys(self) -> KeysView[str]:
        """Parameter bounds keys."""
        return self._mapping.keys()

    def values(
        self,
    ) -> ValuesView[PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]]:
        """Parameter bounds values."""
        return self._mapping.values()

    def items(
        self,
    ) -> ItemsView[str, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]]:
        """Parameter bounds items."""
        return self._mapping.items()

    def __or__(self, other: Any) -> ParamBounds[Array]:
        """Combine two ParamBounds instances."""
        if not isinstance(other, ParamBounds):
            raise NotImplementedError

        pbs = type(self)(**self._mapping)

        for k, v in other.items():
            if k not in pbs or isinstance(v, PriorBounds):
                pbs._mapping[k] = v
                continue
            elif isinstance((sv := pbs._mapping[k]), PriorBounds):
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
    # Flat

    def flatitems(
        self,
    ) -> Iterable[tuple[tuple[str] | tuple[str, str], PriorBounds[Array]]]:
        """Flattened items."""
        for name, bounds in self.items():
            if isinstance(bounds, PriorBounds):
                yield (name,), bounds
            else:
                for subname, subbounds in bounds.items():
                    yield (name, subname), subbounds

    def flatkeys(self) -> tuple[tuple[str] | tuple[str, str], ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatitems())

    def flatvalues(self) -> tuple[PriorBounds[Array], ...]:
        """Flattened values."""
        return tuple(v for _, v in self.flatitems())


class ParamBoundsField(Generic[Array]):
    """Dataclass descriptor for parameter bounds.

    Parameters
    ----------
    default : ParamBounds or Mapping or None, optional
        The default parameter bounds, by default `None`. If `None`, there are no
        default bounds and the parameter bounds must be specified in the Model
        constructor. If not a `ParamBounds` instance, it will be converted to
        one.

    Notes
    -----
    See https://docs.python.org/3/library/dataclasses.html for more information
    on descriptor-typed fields for dataclasses.
    """

    def __init__(
        self,
        default: ParamBounds[Array]
        | Mapping[str, PriorBounds[Array] | Mapping[str, PriorBounds[Array]]]
        | None = None,
    ) -> None:
        self._default: ParamBounds[Array] | None
        self._default = ParamBounds(default) if default is not None else None

    def __set_name__(self, _: type, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: object | None, _: type | None) -> ParamBounds[Array]:
        if obj is not None:
            val: ParamBounds[Array] = getattr(obj, self._name)
            return val

        default = self._default
        if default is not None:
            return default
        else:
            raise AttributeError(f"no default value for {self._name}")

    def __set__(self, obj: object, value: ParamBounds[Array]) -> None:
        dv: ParamBounds[Array] = ParamBounds(self._default or {})
        value = ParamBounds(value)
        object.__setattr__(obj, self._name, dv | value)
