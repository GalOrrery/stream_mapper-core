"""Core feature."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import EllipsisType
from typing import TYPE_CHECKING, TypeGuard, TypeVar

from stream_ml.core.params._base import ParamThingyBase
from stream_ml.core.params.scales.builtin import Identity, ParamScaler
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDict

if TYPE_CHECKING:
    Self = TypeVar("Self", bound="ParamScalers[Array]")  # type: ignore[valid-type]  # noqa: E501

T = TypeVar("T", bound=str | EllipsisType)

__all__: list[str] = []


def _resolve_scaler(b: ParamScaler[Array] | None) -> ParamScaler[Array]:
    """Resolve a bound to a ParamScaler instance."""
    return Identity() if b is None else b


class ParamScalersBase(ParamThingyBase[T, ParamScaler[Array]]):
    """Base class for parameter bounds."""

    _Object = ParamScaler

    @staticmethod
    def _prepare_freeze(
        xs: dict[
            str | T, ParamScaler[Array] | None | Mapping[str, ParamScaler[Array] | None]
        ],
        /,
    ) -> dict[str | T, ParamScaler[Array] | FrozenDict[str, ParamScaler[Array]]]:
        """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
        return {
            k: (
                _resolve_scaler(v)
                if not isinstance(v, Mapping)
                else FrozenDict({kk: _resolve_scaler(vv) for kk, vv in v.items()})
            )
            for k, v in xs.items()
        }


class ParamScalers(ParamScalersBase[str, Array]):
    """A frozen (hashable) dictionary of parameters."""


class IncompleteParamScalers(ParamScalersBase[EllipsisType, Array]):
    """An incomplete parameter bounds."""

    @property
    def is_completable(self) -> bool:
        """Check if the parameter bounds are complete."""
        return is_completable(self)

    def complete(
        self, coord_names: tuple[str, ...] | Iterator[str]
    ) -> ParamScalers[Array]:
        """Complete the parameter bounds.

        Parameters
        ----------
        coord_names : tuple of str
            The coordinate names.

        Returns
        -------
        ParamScalers
            The completed parameter bounds.
        """
        m: dict[str, ParamScaler[Array] | FrozenDict[str, ParamScaler[Array]]] = {}

        for k, v in self.items():
            if isinstance(k, str):
                m[k] = v
                continue
            elif not isinstance(v, Mapping):
                msg = f"incomplete parameter bounds must be a mapping, not {v}"
                raise TypeError(msg)

            for cn in coord_names:
                m[cn] = v

        return ParamScalers(m, __unsafe_skip_copy__=True)


def is_completable(
    pbs: Mapping[str | T, ParamScaler[Array] | Mapping[str, ParamScaler[Array] | None]],
    /,
) -> TypeGuard[Mapping[str, ParamScaler[Array] | Mapping[str, ParamScaler[Array]]]]:
    """Check if parameter names are complete."""
    return all(not isinstance(k, EllipsisType) for k in pbs)
