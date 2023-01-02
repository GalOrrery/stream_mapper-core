"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
from torch import nn
from torch.distributions import MultivariateNormal as TorchMultivariateNormal

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import Params
from stream_ml.pytorch.stream.base import StreamModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array

__all__: list[str] = []


_log2pi = xp.log(xp.asarray(2 * xp.pi))


@dataclass(unsafe_hash=True)
class MultivariateNormal(StreamModel):
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

    n_features: int = 50
    n_layers: int = 3

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the param_names
        expect = (
            ("weight",),
            *((c, p) for c in self.coord_names for p in ("mu", "sigma")),
        )
        if self.param_names.flats != expect:
            msg = f"Expected param_names.flats={expect}, got {self.param_names.flats}"
            raise ValueError(msg)

        # Validate the param_bounds
        if self.param_bounds.flatkeys() != expect:
            msg = (
                f"Expected param_bounds.flatkeys()={expect}, "
                f"got {self.param_bounds.flatkeys()}"
            )
            raise ValueError(msg)

        # Define the layers of the neural network:
        # Total: in (phi) -> out (fraction, *mean, *sigma)
        ndim = len(self.param_names) - 1

        self.layers = nn.Sequential(
            nn.Linear(1, self.n_features),
            nn.Tanh(),
            *functools.reduce(
                operator.add,
                (
                    (nn.Linear(self.n_features, self.n_features), nn.Tanh())
                    for _ in range(self.n_layers - 2)
                ),
            ),
            nn.Linear(self.n_features, 1 + 2 * ndim),
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : Data[Array]
            Data (phi1, phi2, ...).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        eps = xp.finfo(pars[("weight",)].dtype).eps  # TODO: or tiny?
        datav = data[self.coord_names].array

        lik = TorchMultivariateNormal(
            xp.hstack([pars[c, "mu"] for c in self.coord_names]),
            xp.diag_embed(xp.hstack([pars[c, "sigma"] for c in self.coord_names]) ** 2),
        ).log_prob(datav)

        return xp.log(xp.clip(pars[("weight",)], eps)) + lik[:, None]

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("weight",)])  # 100%
        # Bounds
        for bounds in self.param_bounds.flatvalues():
            lnp += bounds.logpdf(pars, data, self, lnp)
        return lnp

    # ========================================================================
    # ML

    def forward(self, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data[Array]
            Input. Only uses the first argument.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        nn = self._forward_prior(self.layers(data[self.indep_coord_name]), data)

        # Call the prior to limit the range of the parameters
        # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            nn = prior(nn, data, self)

        return nn


##############################################################################


@dataclass(unsafe_hash=True)
class MultivariateMissingNormal(MultivariateNormal):  # (MultivariateNormal)
    """Multivariate Normal with missing data."""

    n_features: int = 36
    n_layers: int = 4

    def ln_likelihood_arr(
        self,
        pars: Params[Array],
        data: Data[Array],
        *,
        mask: Array | None = None,
        **kwargs: Array,
    ) -> Array:
        """Negative log-likelihood.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.
        mask : Array
            Mask.
        **kwargs : Array
            Additional arguments.
        """
        datav = data[self.coord_names].array
        mu = xp.hstack([pars[c, "mu"] for c in self.coord_names])
        sigma = xp.hstack([pars[c, "sigma"] for c in self.coord_names])

        if mask is None:
            mask = xp.ones_like(datav)

        # misc
        eps = xp.finfo(datav.dtype).eps  # TODO: or tiny?
        dimensionality = mask.sum(dim=1, keepdim=True)  # (N, 1)

        # Data - model
        dmm = mask * (datav - mu)  # (N, 4)

        # Covariance related
        cov = mask * sigma**2  # (N, 4) positive definite  # TODO: add eps
        det = (cov + (1 - mask)).prod(dim=1, keepdims=True)  # (N, 1)

        return xp.log(xp.clip(pars[("weight",)], min=eps)) - 0.5 * (
            dimensionality * _log2pi  # dim of data
            + xp.log(det)
            + (  # TODO: speed up
                dmm[:, None, :]  # (N, 1, 4)
                @ xp.linalg.pinv(xp.diag_embed(cov))  # (N, 4, 4)
                @ dmm[:, :, None]  # (N, 4, 1)
            )[:, :, 0]
        )  # (N, 1)
