"""Gaussian stream model."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_ml.core._core.base import ModelBase
from stream_ml.core.builtin._stats.norm import logpdf, logpdf_gaussian_errors
from stream_ml.core.typing import Array, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params import Params


@dataclass(unsafe_hash=True)
class Normal(ModelBase[Array, NNModel]):
    r"""1D Gaussian with mixture weight.

    :math:`\mathcal{N}(weight, \mu, \sigma)(\phi1)`
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the distribution.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]

        if self.coord_err_names is None:
            return logpdf(
                data[c],
                mpars[c, "mu"],
                self.xp.clip(mpars[c, "sigma"], 1e-10),
                xp=self.xp,
            )

        return logpdf_gaussian_errors(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.clip(mpars[c, "sigma"], 1e-10),
            sigma_o=self.xp.clip(data[self.coord_err_names[0]], 1e-10),
            xp=self.xp,
        )
