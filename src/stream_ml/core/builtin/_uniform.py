"""Built-in background models."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any

from stream_ml.core._core.base import ModelBase
from stream_ml.core.builtin._stats.uniform import logpdf
from stream_ml.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params


@dataclass(unsafe_hash=True)
class Uniform(ModelBase[Array, NNModel]):
    """Uniform background model."""

    net: None = None  # type: ignore[assignment]

    _: KW_ONLY
    require_mask: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()

        # Pre-compute the log-difference, shape (1, F)
        # First need to check that the bound are finite.
        ab_ = self.xp.asarray(tuple(self.coord_bounds.values()))
        if not self.xp.isfinite(ab_).all():
            msg = "a bound of a coordinate is not finite"
            raise ValueError(msg)
        self._a: Array
        self._b: Array
        object.__setattr__(self, "_a", ab_[None, :, 0])
        object.__setattr__(self, "_b", ab_[None, :, 1])

    # ========================================================================
    # Statistics

    def ln_likelihood(
        self,
        mpars: Params[Array],
        /,
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
        # indicator: (N, F)
        if mask is not None:
            indicator = self.xp.squeeze(mask[tuple(self.coord_bounds.keys())].array)
        elif self.require_mask:
            msg = "mask is required"
            raise ValueError(msg)
        else:
            indicator = self.xp.ones((1, len(self.coord_names)), dtype=int)
            # shape (1, F) so that it can broadcast with (N, F)

        if self.coord_err_names is None:
            return (
                indicator
                * logpdf(data[self.coord_names].array, a=self._a, b=self._b, xp=self.xp)
            ).sum(1)

        raise NotImplementedError
