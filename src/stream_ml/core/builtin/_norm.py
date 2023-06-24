"""Gaussian model."""

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

    :math:`\mathcal{N}(weight, \mu, \ln\sigma)(\phi1)`

    Notes
    -----
    The model parameters are:
    - "mu" : mean
    - "ln-sigma" : log-standard deviation
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)
        if self.coord_err_names is not None and len(self.coord_err_names) != 1:
            msg = "Only one coordinate error is supported, e.g ('phi2_err',)"
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
                loc=mpars[c, "mu"],
                sigma=self.xp.exp(mpars[c, "ln-sigma"]),
                xp=self.xp,
            )

        return logpdf_gaussian_errors(
            data[c],
            loc=mpars[c, "mu"],
            sigma=self.xp.exp(mpars[c, "ln-sigma"]),
            sigma_o=self.xp.log(data[self.coord_err_names[0]]),
            xp=self.xp,
        )
