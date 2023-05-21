"""Parameter."""

from typing import Generic, Protocol, final

from stream_ml.core.params._collection import ModelParameters
from stream_ml.core.typing import Array

__all__: list[str] = []


class SupportsCoordNames(Protocol):
    """Protocol for coordinate names."""

    coord_names: tuple[str, ...]


@final
class ModelParametersField(Generic[Array]):
    """Dataclass descriptor for parameters."""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = "_" + name

    def __get__(
        self, obj: object | None, obj_cls: type | None
    ) -> ModelParameters[Array]:
        if obj is not None:
            val: ModelParameters[Array] = getattr(obj, self._name)
            return val

        msg = f"no default value for {self._name}"
        raise AttributeError(msg)

    def __set__(self, model: SupportsCoordNames, value: ModelParameters[Array]) -> None:
        # TODO: allow input to be many types, cast to Params
        # TODO! validation
        object.__setattr__(model, self._name, ModelParameters[Array](value))
