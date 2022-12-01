"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from typing import TYPE_CHECKING, ClassVar

# THIRD-PARTY
import torch as xp
import torch.nn as nn

# LOCAL
from stream_ml.pytorch.stream.base import StreamModel
from stream_ml.pytorch.utils import within_bounds
from stream_ml.pytorch.utils.funcs import norm_logpdf, sigmoid

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT, ParsT

__all__: list[str] = []


class SingleGaussianStreamModel(StreamModel):
    """Stream Model.

    Parameters
    ----------
    n_layers : int, optional
        Number of hidden layers, by default 3.
    hidden_features : int, optional
        Number of hidden features, by default 50.
    sigma_upper_limit : float, optional keyword-only
        Upper limit on sigma, by default 0.3.
    fraction_upper_limit : float, optional keyword-only
        Upper limit on fraction, by default 0.45.s
    """

    _param_names: ClassVar[dict[str, int]] = {"fraction": 1, "mu": 1, "sigma": 1}

    def __init__(
        self,
        n_layers: int = 3,
        hidden_features: int = 50,
        *,
        fraction_limit: tuple[float, float] = (0.0, 0.45),
        sigma_limit: tuple[float, float] = (0.0, 0.3),
    ) -> None:
        super().__init__()  # Need to do this first.

        # The priors.
        self.sigma_limit = sigma_limit
        self.fraction_limit = fraction_limit

        # Define the layers of the neural network:
        # Total: in (phi) -> out (fraction, mean, sigma)
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.Tanh(),
            *functools.reduce(
                operator.add,
                (
                    (nn.Linear(hidden_features, hidden_features), nn.Tanh())
                    for _ in range(n_layers - 2)
                ),
            ),
            nn.Linear(hidden_features, 3),
        )

    # ========================================================================
    # Statistics

    def ln_likelihood(self, pars: ParsT, data: DataT) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : DataT
            Data (phi1, phi2).
        """
        return norm_logpdf(
            data["phi2"], mu=pars["mu"], sigma=pars["sigma"], amp=pars["fraction"]
        )

    def ln_prior(self, pars: ParsT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars["fraction"])  # 100%

        # Bounds
        lnp[~within_bounds(pars["fraction"], *self.fraction_limit)] = -xp.inf
        lnp[~within_bounds(pars["sigma"], *self.sigma_limit)] = -xp.inf

        return lnp

    # ========================================================================
    # ML

    def forward(self, *args: Array) -> Array:
        """Forward pass.

        Parameters
        ----------
        args : Array
            Input. Only uses the first argument.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        pred = self.layers(args[0])

        fraction = sigmoid(pred[:, 0], *self.fraction_limit)
        mean = pred[:, 1]
        sigma = sigmoid(pred[:, 2], *self.sigma_limit)

        return xp.vstack([fraction, mean, sigma]).T
