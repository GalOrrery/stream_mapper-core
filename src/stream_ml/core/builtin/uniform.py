"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

from stream_ml.core.base import ModelBase
from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.prior.bounds import NoBounds
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.scale.utils import rescale

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params
    from stream_ml.core.typing import ArrayNamespace


@dataclass(unsafe_hash=True)
class Uniform(ModelBase[Array, NNModel]):
    """Uniform background model."""

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField((WEIGHT_NAME,))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](
        {WEIGHT_NAME: NoBounds(param_name=(WEIGHT_NAME,))}
        # TODO: eps, not 1e-10
    )
    require_mask: bool = False

    def __post_init__(self, array_namespace: ArrayNamespace[Array]) -> None:
        super().__post_init__(array_namespace=array_namespace)

        # Pre-compute the log-difference, shape (1, F)
        # First need to check that the bound are finite.
        _bma = []
        for k, (a, b) in self.coord_bounds.items():
            if not self.xp.isfinite(a) or not self.xp.isfinite(b):
                msg = f"a bound of coordinate {k} is not finite"
                raise ValueError(msg)
            _bma.append(b - a)
        self._ln_diffs = self.xp.log(self.xp.asarray(_bma)[None, :])

    def _net_init_default(self) -> NNModel:
        return self.xpnn.Identity()

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
        data: Data[Array],
        *,
        mask: Data[Array] | None = None,
        **kwargs: Any,
    ) -> Array:
        """Log-likelihood of the background.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Labelled data.

        mask : (N, 1) Data[Array[bool]], keyword-only
            Data availability. `True` if data is available, `False` if not.
            Should have the same keys as `data`.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        if mask is not None:
            indicator = mask[tuple(self.coord_bounds.keys())].array
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = self.xp.ones_like(self._ln_diffs, dtype=int)
            # shape (1, F) so that it can broadcast with (N, F)

        ln_wgt = self.xp.log(self.xp.clip(mpars[(WEIGHT_NAME,)], 1e-10))
        return ln_wgt - (indicator * self._ln_diffs).sum(1)[:, None]
