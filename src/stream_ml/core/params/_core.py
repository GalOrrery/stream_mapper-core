"""Parameter."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Generic, cast

from stream_ml.core.params.bounds._base import PriorBounds  # noqa: TCH001
from stream_ml.core.params.scaler._builtin import Identity
from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.params.scaler._api import ParamScaler


@dataclass(frozen=True)
class ModelScalerField:
    def __set_name__(self, owner: type, name: str) -> None:
        self._name: str
        object.__setattr__(self, "_name", "_" + name)

    def __get__(
        self, model: ModelParameter[Array] | None, model_cls: Any
    ) -> ParamScaler[Array]:
        if model is None:
            msg = f"no default value for {self._name!r}."
            raise AttributeError(msg)

        return cast("ParamScaler[Array]", getattr(model, self._name))

    def __set__(
        self, model: ModelParameter[Array], value: ParamScaler[Array] | None
    ) -> None:
        object.__setattr__(
            model, self._name, value if value is not None else Identity()
        )


@dataclass(frozen=True)
class ModelParameter(Generic[Array]):
    _: KW_ONLY
    name: str | None = None
    bounds: PriorBounds[Array]
    scaler: ModelScalerField = ModelScalerField()
