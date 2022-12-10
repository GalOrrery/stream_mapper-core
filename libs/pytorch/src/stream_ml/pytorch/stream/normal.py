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
from stream_ml.core._typing import BoundsT
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.utils.hashdict import FrozenDict
from stream_ml.pytorch.prior.bounds import NoBounds, PriorBounds, SigmoidBounds
from stream_ml.pytorch.stream.base import StreamModel
from stream_ml.pytorch.utils.sigmoid import ColumnarScaledSigmoid

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class Normal(StreamModel):
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
        if self.param_names != ("mixparam", (cn, ("mu", "sigma"))):
            raise ValueError(
                "param_names must be ('mixparam', (<coordinate>, ('mu', 'sigma')))."
            )

        # Validate the param_bounds
        for pn in self.param_names.flats:
            if not self.param_bounds.__contains__(pn):
                raise ValueError(f"param_bounds must contain {pn} (unflattened).")
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
            nn.Linear(self.n_features, 3),
        )
        self.output_scaling = ColumnarScaledSigmoid(
            tuple(range(len(self.param_names.flat))),
            tuple(v.as_tuple() for v in self.param_bounds.flatvalues()),
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: BoundsT = (-inf, inf),
        mixparam_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 1),  # noqa: B008
        mu_bounds: PriorBounds | BoundsT = NoBounds(),  # noqa: B008
        sigma_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 0.3),  # noqa: B008
    ) -> Normal:
        """Create a Normal from a simpler set of inputs.

        Parameters
        ----------
        n_features : int, optional
            Number of features, by default 50.
        n_layers : int, optional
            Number of layers, by default 3.

        coord_name : str, keyword-only
            Coordinate name.
        coord_bounds : BoundsT, optional keyword-only
            Coordinate bounds.
        mixparam_bounds : PriorBounds | BoundsT, optional keyword-only
            Bounds on the mixture parameter.
        mu_bounds : PriorBounds | BoundsT, optional keyword-only
            Bounds on the mean.
        sigma_bounds : PriorBounds | BoundsT, optional keyword-only
            Bounds on the standard deviation.

        Returns
        -------
        Normal
        """
        return cls(
            n_features=n_features,
            n_layers=n_layers,
            coord_names=(coord_name,),
            param_names=ParamNames(("mixparam", (coord_name, ("mu", "sigma")))),  # type: ignore[arg-type] # noqa: E501
            coord_bounds=FrozenDict({coord_name: coord_bounds}),  # type: ignore[arg-type] # noqa: E501
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    "mixparam": cls._make_bounds(mixparam_bounds, ("mixparam",)),
                    coord_name: FrozenDict(
                        mu=cls._make_bounds(mu_bounds, (coord_name, "mu")),
                        sigma=cls._make_bounds(sigma_bounds, (coord_name, "sigma")),
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
        lik = TorchNormal(pars[c, "mu"], xp.clip(pars[c, "sigma"], min=1e-10)).log_prob(
            data[c]
        )
        return xp.log(xp.clip(pars[("mixparam",)], min=1e-10)) + lik

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
        lnp = xp.zeros_like(pars[("mixparam",)])  # 100%
        # Bounds
        for bound in self.param_bounds.flatvalues():
            lnp += bound.logpdf(pars, lnp)
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
        return self.output_scaling(self.layers(args[0]))
