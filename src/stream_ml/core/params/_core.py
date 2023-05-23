"""Parameter."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, Any, Generic, cast

from stream_ml.core.params.bounds._base import ParameterBounds  # noqa: TCH001
from stream_ml.core.params.scaler._builtin import Identity
from stream_ml.core.typing import Array, ParamNameTupleOpts

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
    """Model parameter.

    Parameters
    ----------
    bounds : ParameterBounds, optional keyword-only
        The bounds of the parameter, by default :class:`stream_ml.core.ParameterBounds`.
    scaler : ParamScaler, optional keyword-only
        The scaler for the parameter.

    param_name : tuple[str] | tuple[str, str] | None, optional keyword-only
        The name of the parameter in the :class:`stream_ml.core.ModelParameters`
        dict, by default `None`.
    """

    _: KW_ONLY
    bounds: ParameterBounds[Array]
    scaler: ModelScalerField = ModelScalerField()
    param_name: ParamNameTupleOpts | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bounds",
            replace(self.bounds, scaler=self.scaler, param_name=self.param_name),
        )
