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
from torch.distributions.normal import Normal as TorchNormal

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.utils.hashdict import FrozenDict
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.pytorch.stream.base import StreamModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import BoundsT
    from stream_ml.pytorch._typing import Array

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class DoubleGaussian(StreamModel):
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

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)."
            raise ValueError(msg)
        cn = self.coord_names[0]

        # Set the param names  # TODO:

        # Validate the param_names
        if self.param_names != ("weight1", "weight2", (cn, ("mu", "sigma1", "sigma2"))):
            msg = (
                "param_names must be ('weight1', 'weight2', "
                f"({cn}, ('mu', 'sigma1', 'sigma2')))."
            )
            raise ValueError(msg)

        # Validate the param_bounds
        for pn in self.param_names.flats:
            if not self.param_bounds.__contains__(pn):
                msg = f"param_bounds must contain {pn} (unflattened)."
                raise ValueError(msg)
        # TODO: recursively check for all sub-parameters

        # Define the layers of the neural network:
        # Total: in (phi) -> out (fraction, mean, sigma)
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
            nn.Linear(self.n_features, 5),
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: BoundsT,
        weight1_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 0.45),  # noqa: B008
        weight2_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 0.2),  # noqa: B008
        mu_bounds: PriorBounds | BoundsT | None = None,
        sigma1_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 0.3),  # noqa: B008
        sigma2_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 0.3),  # noqa: B008
    ) -> DoubleGaussian:
        """Create a DoubleGaussian from a simpler set of inputs.

        Returns
        -------
        DoubleGaussian
        """
        return cls(
            n_features=n_features,
            n_layers=n_layers,
            coord_names=(coord_name,),
            param_names=ParamNames(  # type: ignore[arg-type]
                (
                    "weight1",
                    "weight2",
                    (coord_name, ("mu_1", "sigma1", "mu_2", "sigma2")),
                )
            ),
            coord_bounds=FrozenDict(  # type: ignore[arg-type]
                {coord_name: coord_bounds}
            ),
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    "weight1": cls._make_bounds(weight1_bounds, ("weight1",)),
                    "weight2": cls._make_bounds(weight2_bounds, ("weight2",)),
                    coord_name: FrozenDict(
                        mu=cls._make_bounds(mu_bounds, (coord_name, "mu")),
                        sigma1=cls._make_bounds(sigma1_bounds, (coord_name, "sigma1")),
                        sigma2=cls._make_bounds(sigma2_bounds, (coord_name, "sigma2")),
                    ),
                }
            ),
        )

    # ========================================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]
        eps = xp.finfo(pars[("weight",)].dtype).eps  # TOOD: or tiny?

        pre1 = xp.log(xp.clip(pars[("weight1",)], min=eps))
        lik1 = TorchNormal(pars[c, "mu"], xp.clip(pars[c, "sigma1"], min=eps)).log_prob(
            data[c]
        )

        pre2 = xp.log(xp.clip(pars[("weight2",)], min=eps))
        lik2 = TorchNormal(pars[c, "mu"], xp.clip(pars[c, "sigma2"], min=eps)).log_prob(
            data[c]
        )
        return xp.logaddexp(pre1 + lik1, pre2 + lik2)

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("weight1",)])  # 100%
        # Bounds
        for bound in self.param_bounds.flatvalues():
            lnp += bound.logpdf(pars, data, self, lnp)

        # TODO! as embedded Priors
        # sigma2 > sigma1 & weight1 < fraction_1
        c = self.coord_names[0]
        lnp[pars[c, "sigma2"] < pars[c, "sigma1"]] = -xp.inf
        lnp[pars[("weight1",)] < pars[("weight2",)]] = -xp.inf

        # Fractions sum to less than 1
        lnp[(pars[("weight1",)] + pars[("weight2",)]) > 1] = -xp.inf

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
        res = self.layers(data)

        out = res.clone()  # avoid in-place operations
        c = self.coord_names[0]
        pns = self.param_names.flat

        # TODO: there's probably a better way to do this, without relu, which
        # squashes values to 0. But it's not a big deal, since the other value
        # can still be very negative.

        # Ensure weight1 > weight2
        im1, im2 = pns.index("weight1"), pns.index("weight2")
        out[:, im1] = res[:, im2] + xp.relu(res[:, im1])  # weight1: [weight2, inf)

        # TODO: ensure weight1 + weight2 < 1

        # Ensure sigma2 > sigma1\
        is1, is2 = pns.index(f"{c}_sigma1"), pns.index(f"{c}_sigma2")
        out[:, is2] = res[:, is1] + xp.relu(res[:, is2])  # sigma2: [sigma1, inf)

        # use scaled sigmoid to ensure things are in bounds.
        return self._forward_prior(out, data)
