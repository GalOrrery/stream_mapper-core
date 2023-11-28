"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from stream_mapper.core.typing import Array, NNModel
from stream_mapper.core.typing._nn import NNModelProtocol
from stream_mapper.core.utils.sentinel import MISSING, MissingT

if TYPE_CHECKING:
    from stream_mapper.core import ModelAPI as Model


OtherValue = TypeVar("OtherValue")


@dataclass(frozen=True)
class NNField(Generic[NNModel, OtherValue]):
    """Dataclass descriptor for attached nn.

    Parameters
    ----------
    default : NNModel | MISSING, optional
        Default value, by default ``MISSING``.

        - `NNModel` : a value.
        - `None` : defer setting a value until model init.
    """

    default: NNModel | OtherValue | MissingT = MISSING
    _name: str = field(init=False, repr=False, compare=False)

    def __set_name__(self, owner: type, name: str) -> None:
        object.__setattr__(self, "_name", "_" + name)

    def __get__(
        self: NNField[NNModel, OtherValue],
        model: Model[Array, NNModel] | None,
        model_cls: Any,
    ) -> NNModel | OtherValue:
        if model is not None:
            return cast("NNModel", getattr(model, self._name))
        elif self.default is MISSING:
            msg = f"no default value for field {self._name!r}."
            raise AttributeError(msg)
        return self.default

    def __set__(self, model: Model[Array, NNModel], value: NNModel | Any) -> None:
        if not isinstance(
            value,
            (NNModelProtocol,)
            + ((type(self.default),) if self.default is not MISSING else ()),
        ):
            msg = "must provide a wrapped neural network."
            raise TypeError(msg)

        object.__setattr__(model, self._name, value)
