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
from torch import nn
from torch.distributions.normal import Normal as TorchNormal

# LOCAL
from stream_ml.core._typing import BoundsT
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.prior.bounds import NoBounds
from stream_ml.core.utils.hashdict import FrozenDict
from stream_ml.pytorch.prior.bounds import PriorBounds, SigmoidBounds
from stream_ml.pytorch.stream.base import StreamModel

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array

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

    param_names: ParamNamesField = ParamNamesField(("weight", (..., ("mu", "sigma"))))

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)
        cn = self.coord_names[0]

        # Validate the param_names
        # If the param_names are an IncompleteParamNames, then this will
        # complete them.
        if self.param_names != ("weight", (cn, ("mu", "sigma"))):
            msg = (
                f"param_names must be ('weight', ({cn}, ('mu', 'sigma'))),"
                f"got {self.param_names}"
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
            nn.Linear(self.n_features, 3),
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: BoundsT = (-inf, inf),
        weight_bounds: PriorBounds | BoundsT = SigmoidBounds(0, 1),  # noqa: B008
        mu_bounds: PriorBounds | BoundsT | None | NoBounds = None,
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
        weight_bounds : PriorBounds | BoundsT, optional keyword-only
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
            param_names=ParamNames(("weight", (coord_name, ("mu", "sigma")))),  # type: ignore[arg-type] # noqa: E501
            coord_bounds=FrozenDict({coord_name: coord_bounds}),  # type: ignore[arg-type] # noqa: E501
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    "weight": cls._make_bounds(weight_bounds, ("weight",)),
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
        lik = TorchNormal(pars[c, "mu"], xp.clip(pars[c, "sigma"], min=eps)).log_prob(
            data[c]
        )
        return xp.log(xp.clip(pars[("weight",)], min=eps)) + lik

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
        for bound in self.param_bounds.flatvalues():
            lnp += bound.logpdf(pars, data, self, lnp)
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
