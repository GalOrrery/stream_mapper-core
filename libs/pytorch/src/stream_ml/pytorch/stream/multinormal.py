"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
import torch.nn as nn
from torch.distributions import MultivariateNormal as TorchMultivariateNormal

# LOCAL
from stream_ml.core.utils.params import Params
from stream_ml.pytorch.stream.base import StreamModel
from stream_ml.pytorch.utils import within_bounds
from stream_ml.pytorch.utils.sigmoid import ColumnarScaledSigmoid

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT

__all__: list[str] = []


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
        # TODO: this should be automatic
        expect = (
            "mixparam",
            *((c, p) for c in self.coord_names for p in ("mu", "sigma")),
        )
        if self.param_names != expect:
            raise ValueError(f"Expected param_names={expect}, got {self.param_names}")

        # Validate the param_bounds
        # TODO!

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
            ColumnarScaledSigmoid(
                (0, *range(ndim + 1, 2 * ndim + 1)),
                (
                    self.param_bounds[("mixparam",)],
                    *(self.param_bounds[c, "sigma"] for c in self.coord_names),
                ),
            ),
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, *args: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data (phi1, phi2, ...).
        *args : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        lik = TorchMultivariateNormal(
            xp.hstack([pars[c, "mu"] for c in self.coord_names]),
            xp.diag_embed(xp.hstack([pars[c, "sigma"] for c in self.coord_names]) ** 2),
        ).log_prob(xp.hstack(list(data.values()))[:, 1:])

        return xp.log(xp.clip(pars[("mixparam",)], 0)) + lik[:, None]

    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("mixparam",)])  # 100%
        # Bounds
        for names, bounds in self.param_bounds.flatitems():
            lnp[~within_bounds(pars[names], *bounds)] = -xp.inf
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
        return pred


##############################################################################


@dataclass(unsafe_hash=True)
class MultivariateMissingNormal(MultivariateNormal):
    """Multivariate Normal with missing data."""

    def ln_likelihood_arr(
        self, pars: Params[Array], data: DataT, *args: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data (phi1, phi2, ...).
        *args : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        datav = xp.hstack(list(data.values()))[:, 1:]

        mu = xp.hstack([pars[c, "mu"] for c in self.coord_names])
        cov = xp.diag_embed(
            xp.hstack([pars[c, "sigma"] for c in self.coord_names]) ** 2
        )
        inv_cov_total = xp.inverse(cov)  # nstar x 5 x 5
        cov_avail = args[0]  # TODO: as a kwarg
        # nstar x 5 x 5: diagonal matrix of zeros and infs for each star.
        # Infs where there is no data.

        data_minus_model = datav - mu  # n_star x 5
        right_product = xp.einsum("ijk,ik->ij", inv_cov_total, data_minus_model)
        right_product[~cov_avail] = xp.tensor([0.0]).float()
        exp_arg = -0.5 * xp.sum(data_minus_model * right_product, dim=1)  # n_star
        det_arr = xp.zeros(len(cov))
        for i in range(len(det_arr)):
            det_arr[i] = xp.prod(cov[i, cov_avail[i, :], cov_avail[i, :]])

        pref = xp.sqrt(((2.0 * xp.pi) ** datav.shape[1]) * det_arr)  # n_star

        return xp.log(xp.clip(pars[("mixparam",)], 0)) + exp_arg - xp.log(pref)
