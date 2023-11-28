"""Parameters."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from collections.abc import Iterable, Mapping
from dataclasses import replace
from itertools import chain
from typing import TYPE_CHECKING, Any, overload

from stream_mapper.core.params._core import ModelParameter
from stream_mapper.core.setup_package import PACK_PARAM_JOIN
from stream_mapper.core.typing import Array
from stream_mapper.core.utils.cached_property import cached_noargmethod
from stream_mapper.core.utils.frozen_dict import FrozenDict

if TYPE_CHECKING:
    from stream_mapper.core.typing import ParamNameAllOpts, ParamNameTupleOpts


LEN_NAME_TUPLE: int = 2


class ModelParameters(
    FrozenDict[str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]]
):
    """Param dictionary."""

    def __init__(
        self,
        m: Mapping[
            str, ModelParameter[Array] | Mapping[str, ModelParameter[Array]]
        ] = {},
        /,
        **kwargs: ModelParameter[Array] | Mapping[str, ModelParameter[Array]],
    ) -> None:
        # Shortcut if `m` is ModelParameters and there's no kwargs
        if isinstance(m, ModelParameters) and not kwargs:
            super().__init__(m._dict, __unsafe_skip_copy__=True)
            return

        # Freeze sub-dicts
        d: dict[
            str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]
        ] = {}

        for k, v in chain(m.items(), kwargs.items()):
            if not isinstance(v, Mapping):
                d[k] = replace(v, param_name=(k,))
            else:
                d[k] = FrozenDict[str, ModelParameter[Array]](
                    {k2: replace(v2, param_name=(k, k2)) for k2, v2 in v.items()}
                )

        super().__init__(d, __unsafe_skip_copy__=True)

    # =========================================================================
    # Mapping

    @overload
    def __getitem__(
        self, key: str
    ) -> ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]:
        ...

    @overload
    def __getitem__(
        self, key: ParamNameTupleOpts
    ) -> ModelParameter[Array]:  # Flat keys
        ...

    def __getitem__(
        self, key: ParamNameAllOpts
    ) -> ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]:
        if isinstance(key, str):
            value = self._dict[key]
        elif len(key) == 1:
            value = self._dict[key[0]]
        elif len(key) == LEN_NAME_TUPLE:
            cm = self._dict[key[0]]
            if not isinstance(cm, Mapping):
                raise KeyError(str(key))
            value = cm[key[1]]
        else:
            raise KeyError(str(key))
        return value

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
    # Freeze

    def unfreeze(  # type: ignore[override]
        self,
    ) -> dict[str, ModelParameter[Array] | dict[str, ModelParameter[Array]]]:
        """Unfreeze the parameters."""
        return {
            k: (v if not isinstance(v, Mapping) else dict(v)) for k, v in self.items()
        }

    # =========================================================================
    # Flats
    # Tuple keys are used to access the parameters.

    @cached_noargmethod
    def flatsitems(
        self,
    ) -> tuple[tuple[ParamNameTupleOpts, ModelParameter[Array]], ...]:
        """Flattened items."""
        return tuple(_flats_iter(self))

    @cached_noargmethod
    def flatskeys(self) -> tuple[ParamNameTupleOpts, ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatsitems())

    @cached_noargmethod
    def flatsvalues(self) -> tuple[ModelParameter[Array], ...]:
        """Flattened values."""
        return tuple(v for _, v in self.flatsitems())

    # =========================================================================
    # Flat

    @cached_noargmethod
    def flatitems(self) -> tuple[tuple[str, ModelParameter[Array]], ...]:
        """Flat items."""
        return tuple((PACK_PARAM_JOIN.join(k), v) for k, v in self.flatsitems())

    @cached_noargmethod
    def flatkeys(self) -> tuple[str, ...]:
        """Flat keys."""
        return tuple(k for k, _ in self.flatitems())

    def flatvalues(self) -> tuple[ModelParameter[Array], ...]:
        """Flat values."""
        return self.flatsvalues()


def _flats_iter(
    params: ModelParameters[Array], /
) -> Iterable[
    tuple[tuple[str], ModelParameter[Array]]
    | tuple[tuple[str, str], ModelParameter[Array]]
]:
    for k, v in params.items():
        if not isinstance(v, Mapping):
            yield (k,), v
        else:
            for kk, vv in v.items():
                yield (k, kk), vv
