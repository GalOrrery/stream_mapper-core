"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from typing import TYPE_CHECKING, ClassVar

# THIRD-PARTY
import torch as xp
import torch.nn as nn
from torch.distributions.normal import Normal

# LOCAL
from stream_ml.sigmoid import ColumnarScaledSigmoid
from stream_ml.stream.base import StreamModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml._typing import Array, DataT, ParsT

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

    param_names: ClassVar[dict[str, int]] = {"fraction": 0, "mu": 0, "sigma": 0}

    def __init__(
        self,
        n_layers: int = 3,
        hidden_features: int = 50,
        *,
        fraction_upper_limit: float = 0.45,
        sigma_upper_limit: float = 0.3,
    ) -> None:
        super().__init__()  # Need to do this first.

        # The priors.
        self.sigma_upper_limit = sigma_upper_limit
        self.fraction_upper_limit = fraction_upper_limit

        # Define the layers of the neural network:
        # Total: in (phi) -> out (fraction, mean, sigma)
        self.layers = nn.Sequential(
            nn.Linear(1, hidden_features),
            *functools.reduce(
                operator.add, ((nn.Linear(hidden_features, hidden_features), nn.Tanh()) for _ in range(n_layers - 2))
            ),
            nn.Linear(hidden_features, 3),
            ColumnarScaledSigmoid((0, 2), ((0, fraction_upper_limit), (0, sigma_upper_limit))),  # fraction, sigma
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

        Returns
        -------
        Array
        """
        return xp.log(xp.clamp(pars["fraction"], min=0)) + Normal(pars["mu"], xp.clamp(pars["sigma"], min=0)).log_prob(
            data["phi2"]
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
        # Bound fraction in [0, <upper limit>]
        if pars["fraction"] < 0 or self.fraction_upper_limit < pars["fraction"]:
            return -xp.asarray(xp.inf)
        # Bound sigma in [0, <upper limit>]
        elif pars["sigma"] < 0 or self.sigma_upper_limit < pars["sigma"]:
            return -xp.asarray(xp.inf)

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
        return self.layers(args[0])
