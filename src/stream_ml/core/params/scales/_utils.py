"""Utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stream_ml.core.params._core import freeze_params, set_param

__all__: list[str] = []

if TYPE_CHECKING:
    from stream_ml.core._base import ModelBase
    from stream_ml.core.multi import ModelsBase
    from stream_ml.core.params import Params
    from stream_ml.core.typing import Array, NNModel
    from stream_ml.core.utils.frozen_dict import FrozenDict


def scale_params(
    model: ModelBase[Array, NNModel] | ModelsBase[Array, NNModel], mpars: Params[Array]
) -> Params[Array]:
    """Rescale the parameters to the model's scale."""
    pars: dict[str, Any | FrozenDict[str, Any]] = {}
    for kp in model.param_names.flats:
        v = model.param_scalers[kp].transform(mpars[kp])
        set_param(pars, kp, v)
    return freeze_params(pars)
