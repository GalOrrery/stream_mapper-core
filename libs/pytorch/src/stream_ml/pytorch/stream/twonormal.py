"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
import torch.nn as nn
from torch.distributions.normal import Normal as TorchNormal

# LOCAL
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.utils.hashdict import FrozenDict
from stream_ml.pytorch.stream.base import StreamModel
from stream_ml.pytorch.utils import within_bounds
from stream_ml.pytorch.utils.sigmoid import ColumnarScaledSigmoid

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT

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
            raise ValueError("Only one coordinate is supported, e.g ('phi2',).")
        cn = self.coord_names[0]

        # Validate the param_names
        if self.param_names != (
            "mixparam1",
            "mixparam2",
            (cn, ("mu", "sigma1", "sigma2")),
        ):
            raise ValueError(
                "param_names must be ('mixparam1', 'mixparam2', (<coordinate>, "
                "('mu', 'sigma1', 'sigma2')))."
            )

        # Validate the param_bounds
        for pn in self.param_names.flats:
            if not self.param_bounds.__contains__(pn):
                raise ValueError(f"param_bounds must contain {pn} (unflattened).")
        # TODO: recursively check for all sub-parameters
        # [("mixparam", (0.0, 1.0)), ("mu", (-xp.inf, xp.inf)), ("sigma", (0.0, 0.3))]

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
        self.output_scaling = ColumnarScaledSigmoid(
            tuple(range(len(self.param_names.flat))),
            tuple(self.param_bounds.flatvalues()),
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: tuple[float, float],
        mixparam1_bounds: tuple[float, float] = (0, 0.45),
        mixparam2_bounds: tuple[float, float] = (0, 0.2),
        mu_bounds: tuple[float, float] = (-inf, inf),
        sigma1_bounds: tuple[float, float] = (0, 0.3),
        sigma2_bounds: tuple[float, float] = (0, 0.3),
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
                    "mixparam1",
                    "mixparam2",
                    (coord_name, ("mu_1", "sigma1", "mu_2", "sigma2")),
                )
            ),
            coord_bounds=FrozenDict(  # type: ignore[arg-type]
                {coord_name: coord_bounds}
            ),
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    "mixparam1": mixparam1_bounds,
                    "mixparam2": mixparam2_bounds,
                    coord_name: FrozenDict(
                        mu=mu_bounds,
                        sigma1=sigma1_bounds,
                        sigma2=sigma2_bounds,
                    ),
                }
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
            Data (phi1, phi2).
        *args : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]

        pre1 = xp.log(xp.clip(pars[("mixparam1",)], min=1e-10))
        lik1 = TorchNormal(
            pars[c, "mu"], xp.clip(pars[c, "sigma1"], min=1e-10)
        ).log_prob(data[c])

        pre2 = xp.log(xp.clip(pars[("mixparam2",)], min=1e-10))
        lik2 = TorchNormal(
            pars[c, "mu"], xp.clip(pars[c, "sigma2"], min=1e-10)
        ).log_prob(data[c])
        return xp.logaddexp(pre1 + lik1, pre2 + lik2)

    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params
            Parameters.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("mixparam1",)])  # 100%
        # Bounds
        for names, bounds in self.param_bounds.flatitems():
            lnp[~within_bounds(pars[names], *bounds)] = -xp.inf

        # sigma2 > sigma1 & mixparam1 < fraction_1
        c = self.coord_names[0]
        lnp[pars[c, "sigma2"] < pars[c, "sigma1"]] = -xp.inf
        lnp[pars[("mixparam1",)] < pars[("mixparam2",)]] = -xp.inf

        # Fractions sum to less than 1
        lnp[(pars[("mixparam1",)] + pars[("mixparam2",)]) > 1] = -xp.inf

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
        res = self.layers(args[0])

        out = res.clone()  # avoid in-place operations
        c = self.coord_names[0]
        pns = self.param_names.flat

        # TODO: there's probably a better way to do this, without relu, which
        # squashes values to 0. But it's not a big deal, since the other value
        # can still be very negative.

        # Ensure mixparam1 > mixparam2
        im1, im2 = pns.index("mixparam1"), pns.index("mixparam2")
        out[:, im1] = res[:, im2] + xp.relu(res[:, im1])  # mixparam1: [mixparam2, inf)

        # TODO: ensure mixparam1 + mixparam2 < 1

        # Ensure sigma2 > sigma1\
        is1, is2 = pns.index(f"{c}_sigma1"), pns.index(f"{c}_sigma2")
        out[:, is2] = res[:, is1] + xp.relu(res[:, is2])  # sigma2: [sigma1, inf)

        # use scaled sigmoid to ensure things are in bounds.
        return self.output_scaling(out)
