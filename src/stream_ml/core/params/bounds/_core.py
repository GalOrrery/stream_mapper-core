"""Core feature."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from types import EllipsisType
from typing import TYPE_CHECKING, TypeGuard, TypeVar

from stream_ml.core.params._base import ParamXsBase
from stream_ml.core.prior.bounds import NoBounds, PriorBounds
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDict

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core.params.names import ParamNames

    Self = TypeVar("Self", bound="ParamBounds[Array]")  # type: ignore[valid-type]
    Object = TypeVar("Object")

T = TypeVar("T", bound=str | EllipsisType)


def _resolve_bound(b: PriorBounds[Array] | None) -> PriorBounds[Array]:
    """Resolve a bound to a PriorBounds instance."""
    return NoBounds() if b is None else b


# ===================================================================


class ParamBoundsBase(ParamXsBase[T, PriorBounds[Array]]):
    """Base class for parameter bounds."""

    _Object = PriorBounds

    @staticmethod
    def _prepare_freeze(
        xs: dict[
            str | T,
            PriorBounds[Array] | None | Mapping[str, PriorBounds[Array] | None],
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
