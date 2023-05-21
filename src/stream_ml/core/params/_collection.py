"""Parameters."""

from __future__ import annotations

__all__: list[str] = []

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast, overload

from stream_ml.core.params._core import ModelParameter
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDict

if TYPE_CHECKING:
    from stream_ml.core.typing import ParamNameAllOpts, ParamNameTupleOpts


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
        # Freeze sub-dicts
        d: dict[str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]] = {
            k: v
            if not isinstance(v, Mapping)
            else FrozenDict[str, ModelParameter[Array]](v)
            for k, v in dict(m, **kwargs).items()
        }
        super().__init__(d, __unsafe_skip_copy__=True)

        # hint cached data

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
            key = cast("tuple[str, str]", key)  # TODO: remove cast
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

    # TODO: cache
    def flatsitems(
        self,
    ) -> tuple[tuple[ParamNameTupleOpts, ModelParameter[Array]], ...]:
        """Flattened items."""
        return tuple(_flats_iter(self))

    # TODO: cache
    def flatskeys(self) -> tuple[ParamNameTupleOpts, ...]:
        """Flattened keys."""
        return tuple(k for k, _ in self.flatsitems())

    # TODO: cache
    def flatsvalues(self) -> tuple[ModelParameter[Array], ...]:
        """Flattened values."""
        return tuple(v for _, v in self.flatsitems())

    # =========================================================================
    # Flat

    # TODO: cache
    def flatitems(self) -> tuple[tuple[str, ModelParameter[Array]], ...]:
        """Flat items."""
        return tuple(("_".join(k), v) for k, v in self.flatsitems())

    # TODO: cache
    def flatkeys(self) -> tuple[str, ...]:
        """Flat keys."""
        return tuple(k for k, _ in self.flatitems())

    # TODO: cache
    def flatvalues(self) -> tuple[ModelParameter[Array], ...]:
        """Flat values."""
        return tuple(v for _, v in self.flatitems())


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
