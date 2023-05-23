"""Core feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from stream_ml.core.typing import Array, NNModel
from stream_ml.core.typing._nn import NNModelProtocol
from stream_ml.core.utils.sentinel import MISSING, MissingT

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core._core.api import Model

    Self = TypeVar("Self", bound="Model[Array, NNModel]")  # type: ignore[valid-type]  # noqa: E501


@dataclass(frozen=True)
class NNField(Generic[NNModel]):
    """Dataclass descriptor for attached nn.

    Parameters
    ----------
    default : NNModel | None, optional
        Default value, by default `None`.

        - `NNModel` : a value.
        - `None` : defer setting a value until model init.
    """

    default: NNModel | MissingT = MISSING

    def __set_name__(self, owner: type, name: str) -> None:
        self._name: str
        object.__setattr__(self, "_name", "_" + name)

    def __get__(self, model: Model[Array, NNModel] | None, model_cls: Any) -> NNModel:
        if model is not None:
            return cast("NNModel", getattr(model, self._name))
        elif self.default is MISSING:
            msg = f"no default value for field {self._name!r}."
            raise AttributeError(msg)
        return self.default

    def __set__(self, model: Model[Array, NNModel], value: NNModel | Any) -> None:
        if not isinstance(value, NNModelProtocol):
            msg = "must provide a wrapped neural network."
            raise TypeError(msg)

        object.__setattr__(model, self._name, value)
