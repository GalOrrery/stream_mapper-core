"""Utilities."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from typing import TYPE_CHECKING

from stream_mapper.core.params._values import freeze_params, set_param

if TYPE_CHECKING:
    from stream_mapper.core import ModelBase, ModelsBase, Params
    from stream_mapper.core.typing import Array, NNModel


def scale_params(
    model: ModelBase[Array, NNModel] | ModelsBase[Array, NNModel], mpars: Params[Array]
) -> Params[Array]:
    """Rescale the parameters to the model's scale."""
    pars: dict[str, Array | dict[str, Array]] = {}
    for kp, p in model.params.flatsitems():
        v: Array = p.scaler.transform(mpars[kp])
        set_param(pars, kp, v)
    return freeze_params(pars)
