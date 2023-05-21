"""Parameter."""

from dataclasses import KW_ONLY, dataclass
from typing import Generic

from stream_ml.core.params.bounds import PriorBounds
from stream_ml.core.params.scaler._api import ParamScaler
from stream_ml.core.typing import Array

__all__: list[str] = []


@dataclass(frozen=True)
class ModelParameter(Generic[Array]):
    _: KW_ONLY
    name: str | None
    bounds: PriorBounds[Array]
    scaler: ParamScaler[Array]
