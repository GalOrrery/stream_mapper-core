"""Core feature."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from types import EllipsisType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)
from typing import Literal as L  # noqa: N817

from stream_ml.core.prior.bounds import NoBounds, PriorBounds
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDict
from stream_ml.core.utils.sentinel import MISSING, Sentinel

if TYPE_CHECKING:
    from stream_ml.core.params.names import ParamNames, ParamNamesField

    Self = TypeVar("Self", bound="ParamBounds[Array]")  # type: ignore[valid-type]

T = TypeVar("T", bound=str | EllipsisType)

__all__: list[str] = []


##############################################################################


def _resolve_bound(b: PriorBounds[Array] | None) -> PriorBounds[Array]:
    """Resolve a bound to a PriorBounds instance."""
    return NoBounds() if b is None else b


def _prepare_freeze(
    xs: dict[
        str | T, PriorBounds[Array] | None | Mapping[str, PriorBounds[Array] | None]
    ],
    /,
) -> dict[str | T, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]]:
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    return {
        k: (
            _resolve_bound(v)
            if not isinstance(v, Mapping)
            else FrozenDict({kk: _resolve_bound(vv) for kk, vv in v.items()})
        )
        for k, v in xs.items()
    }


# ===================================================================


class ParamBoundsBase(
    FrozenDict[str | T, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]],
):
    """Base class for parameter bounds."""

    def __init__(
        self, m: Any = (), /, *, __unsafe_skip_copy__: bool = False, **kwargs: Any
    ) -> None:
        if __unsafe_skip_copy__ and not kwargs:
            return super().__init__(m, __unsafe_skip_copy__=True)

        # Initialize, with validation.
        # TODO: not cast to dict if already a ParamBounds
        pb: dict[str | T, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]]
        pb = _prepare_freeze(dict(m, **kwargs))

        super().__init__(pb, __unsafe_skip_copy__=True)
        return None

    # =========================================================================
    # Mapping

    @overload
    def __getitem__(
        self, key: str | T
    ) -> PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]:
        ...

    @overload
    def __getitem__(self, key: tuple[str]) -> PriorBounds[Array]:  # Flat key
        ...

    @overload
    def __getitem__(self, key: tuple[str, str]) -> PriorBounds[Array]:  # Flat key
        ...

    def __getitem__(
        self, key: str | T | tuple[str] | tuple[str, str]
    ) -> PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]:
        if isinstance(key, tuple):
            if len(key) == 1:  # e.g. ("weight",)
                value = super().__getitem__(key[0])
                if not isinstance(value, PriorBounds):
                    raise KeyError(key)
            else:  # e.g. ("phi2", "mu")
                key = cast("tuple[str, str]", key)  # TODO: remove cast
                v = super().__getitem__(key[0])
                if not isinstance(v, FrozenDict):
                    raise KeyError(key)
                value = v[key[1]]
        else:  # e.g. "weight"
            value = super().__getitem__(key)
        return value  # noqa: RET504

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
    ) -> Iterable[tuple[tuple[str | T] | tuple[str | T, str], PriorBounds[Array]]]:
        """Flattened items."""
        for name, bounds in self.items():
            if isinstance(bounds, PriorBounds):
                yield (name,), bounds
            else:
                for subname, subbounds in bounds.items():
                    yield (name, subname), subbounds

    def flatkeys(self) -> tuple[tuple[str | T] | tuple[str | T, str], ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatitems())

    def flatvalues(self) -> tuple[PriorBounds[Array], ...]:
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


class ParamBounds(ParamBoundsBase[str, Array]):
    """A frozen (hashable) dictionary of parameters."""

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
            if isinstance(pn, str):  # e.g. "weight"
                m[pn] = replace(default, param_name=(pn,))
            else:  # e.g. ("phi2", ("mu", "sigma"))
                m[pn[0]] = FrozenDict(
                    {k: replace(default, param_name=(pn[0], k)) for k in pn[1]}
                )
        return cls(m, __unsafe_skip_copy__=True)

    # =========================================================================
    # Misc

    # TODO: better method name
    def _fixup_param_names(self) -> None:
        """Set the parameter name in the prior bounds."""
        for k, v in self.items():
            if isinstance(v, PriorBounds):
                if v.param_name is None:
                    self._dict[k] = replace(v, param_name=(k,))
                continue

            for kk, vv in v.items():
                if vv.param_name is None:
                    v._dict[kk] = replace(vv, param_name=(k, kk))


class IncompleteParamBounds(ParamBoundsBase[EllipsisType, Array]):
    """An incomplete parameter bounds."""

    @property
    def is_completable(self) -> bool:
        """Check if the parameter bounds are complete."""
        return is_completable(self)

    def complete(
        self, coord_names: tuple[str, ...] | Iterator[str]
    ) -> ParamBounds[Array]:
        """Complete the parameter bounds.

        Parameters
        ----------
        coord_names : tuple of str
            The coordinate names.

        Returns
        -------
        ParamBounds
            The completed parameter bounds.
        """
        m: dict[str, PriorBounds[Array] | FrozenDict[str, PriorBounds[Array]]] = {}

        for k, v in self.items():
            if isinstance(k, str):
                m[k] = v
                continue
            elif not isinstance(v, Mapping):
                msg = f"incomplete parameter bounds must be a mapping, not {v}"
                raise TypeError(msg)

            for cn in coord_names:
                m[cn] = v

        return ParamBounds(m, __unsafe_skip_copy__=True)


def is_completable(
    pbs: Mapping[
        str | T, PriorBounds[Array] | None | Mapping[str, PriorBounds[Array] | None]
    ],
    /,
) -> TypeGuard[Mapping[str, PriorBounds[Array] | Mapping[str, PriorBounds[Array]]]]:
    """Check if parameter names are complete."""
    return all(not isinstance(k, EllipsisType) for k in pbs)


#####################################################################


class SupportsCoordandParamNames(Protocol):
    """Protocol for coordinate names."""

    coord_names: tuple[str, ...]
    param_names: ParamNamesField
    DEFAULT_BOUNDS: ClassVar


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
        | Mapping[
            str | EllipsisType,
            PriorBounds[Array] | None | Mapping[str, PriorBounds[Array] | None],
        ]
        | L[Sentinel.MISSING] = MISSING,
    ) -> None:
        dft: ParamBounds[Array] | IncompleteParamBounds[Array] | L[Sentinel.MISSING]
        if default is MISSING:
            dft = MISSING
        elif isinstance(default, (ParamBounds, IncompleteParamBounds)):
            dft = default
        elif is_completable(default):
            dft = ParamBounds(default)  # e.g. fills in None -> NoBounds
        else:
            dft = IncompleteParamBounds(default)

        self._default: ParamBounds[Array] | IncompleteParamBounds[Array] | L[
            Sentinel.MISSING
        ]
        self._default = dft

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(
        self, obj: object | None, _: type | None
    ) -> ParamBounds[Array] | IncompleteParamBounds[Array]:
        if obj is not None:
            val: ParamBounds[Array] = getattr(obj, self._name)
            return val

        default = self._default
        if default is MISSING:
            msg = f"no default value for {self._name}"
            raise AttributeError(msg)
        return default

    def __set__(
        self,
        model: SupportsCoordandParamNames,
        value: ParamBounds[Array] | IncompleteParamBounds[Array],
    ) -> None:
        if isinstance(value, IncompleteParamBounds):
            value = value.complete(
                c for c in model.coord_names if c in model.param_names.top_level
            )
        else:
            # TODO! make sure minimal copying is done
            value = ParamBounds(value)

        if self._default is not MISSING:
            default = self._default
            if isinstance(default, IncompleteParamBounds):
                default = default.complete(
                    c for c in model.coord_names if c in model.param_names.top_level
                )

            # Don't need to check if the default is completable, since it
            # merged with a complete value made by ``from_names``.
            # Also, it is validated against the model's parameter names.
            # Both these are done in `stream_ml.core.base.Model`.
            value = default | value

        # TODO: can this be done in the param_bounds field?
        # Make parameter bounds
        # 1) Make the default bounds for all parameters.
        # 2) Update from the user-specified bounds.
        # 3) Fix up the names so each bound references its parameter.
        value = (
            ParamBounds.from_names(model.param_names, default=model.DEFAULT_BOUNDS)
            | value
        )
        value._fixup_param_names()
        # Validate param bounds.
        value.validate(model.param_names)

        object.__setattr__(model, self._name, value)
