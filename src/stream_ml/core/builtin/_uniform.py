"""Built-in background models."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

from stream_ml.core._base import ModelBase, NNField
from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.typing import Array, NNModel

__all__: list[str] = []


if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params


@dataclass(unsafe_hash=True)
class Uniform(ModelBase[Array, NNModel]):
    """Uniform background model."""

    net: NNField[NNModel] = NNField(default=None)

    _: KW_ONLY
    param_names: ParamNamesField = ParamNamesField(())
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array]({})
    require_mask: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Pre-compute the log-difference, shape (1, F)
        # First need to check that the bound are finite.
        _bma = []
        for k, (a, b) in self.coord_bounds.items():
            a_, b_ = self.xp.asarray(a), self.xp.asarray(b)
            if not self.xp.isfinite(a_) or not self.xp.isfinite(b_):
                msg = f"a bound of coordinate {k} is not finite"
                raise ValueError(msg)
            _bma.append(b_ - a_)
        self._ln_liks = -self.xp.log(self.xp.asarray(_bma)[None, :])

    def _net_init_default(self) -> NNModel | None:
        return None

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
            indicator = self.xp.ones_like(self._ln_liks, dtype=int)
            # shape (1, F) so that it can broadcast with (N, F)

        return (indicator * self._ln_liks).sum(1)[:, None]
