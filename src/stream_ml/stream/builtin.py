"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING, ClassVar

# THIRD-PARTY
import torch as xp
import torch.nn as nn
from torch import sigmoid

# LOCAL
from stream_ml.funcs import log_of_normal
from stream_ml.stream.base import StreamModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml._typing import Array, DataT, ParsT

__all__: list[str] = []


class SingleGaussianStreamModel(StreamModel):
    """Stream Model."""

    param_names: ClassVar[dict[str, int]] = {"fraction": 0, "phi2_mu": 0, "phi2_sigma": 0}

    def __init__(self, sigma_upper_limit: float = 0.3, fraction_upper_limit: float = 0.45) -> None:
        super().__init__()  # Need to do this first.

        # The priors. # TODO! implement in the Stream/Background models
        self.sigma_upper_limit = sigma_upper_limit
        self.fraction_upper_limit = fraction_upper_limit

        # Define the layers of the neural network:
        # Total: 1 (phi) -> 3 (fraction, mean, sigma)
        self.layers = nn.Sequential(
            nn.Linear(1, 50),  # layer 1: 1 node -> 50 nodes
            nn.Tanh(),
            nn.Linear(50, 50),  # layer 2: 50 node -> 50 nodes
            nn.Tanh(),
            nn.Linear(50, 3),  # layer 3: 50 node -> 3 nodes
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
        return xp.log(pars["fraction"]) + log_of_normal(data["phi2"], pars["phi2_mu"], pars["phi2_sigma"])

    def ln_prior(self, pars: ParsT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        """
        return xp.asarray(1.0)  # TODO: Implement this!

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

        # TODO: Use the the priors from ln_prior!
        fraction = sigmoid(pred[:, 2]) * self.fraction_upper_limit
        mean = pred[:, 0]
        sigma = sigmoid(pred[:, 1]) * self.sigma_upper_limit

        return xp.vstack([fraction, mean, sigma]).T
