"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from dataclasses import KW_ONLY, dataclass
from math import inf
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
import torch.nn as nn
from torch.distributions.normal import Normal as TorchNormal

# LOCAL
from stream_ml.core.utils.hashdict import HashableMap, HashableMapField
from stream_ml.core.utils.params import (
    ParamBounds,
    ParamBoundsField,
    ParamNames,
    Params,
)
from stream_ml.pytorch.stream.base import StreamModel
from stream_ml.pytorch.utils import within_bounds
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
    _: KW_ONLY

    coord_bounds: HashableMapField[str, tuple[float, float]] = HashableMapField()  # type: ignore[assignment]  # noqa: E501
    param_bounds: ParamBoundsField = ParamBoundsField(ParamBounds())
    # [("mixparam", (0.0, 1.0)), ("mu", (-xp.inf, xp.inf)), ("sigma", (0.0, 0.3))]

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            raise ValueError("Only one coordinate is supported, e.g ('phi2',).")
        cn = self.coord_names[0]

        # Validate the param_names
        if self.param_names != ("mixparam", (cn, ("mu", "sigma"))):
            raise ValueError(
                "param_names must be ('sigma', (<coordinate>, ('mu', 'sigma')))."
            )

        # Validate the param_bounds # TODO!
        # for pn in self.param_names:
        #     if pn not in self.param_bounds:
        #         raise ValueError(f"param_bounds must contain {pn}.")
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
            nn.Linear(self.n_features, 3),
            ColumnarScaledSigmoid(
                (0, 2),
                (
                    self.param_bounds[("mixparam",)],
                    self.param_bounds[cn, "sigma"],
                ),
            ),
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: tuple[float, float],
        mixparam_bounds: tuple[float, float] = (0, 1),
        mu_bounds: tuple[float, float] = (-inf, inf),
        sigma_bounds: tuple[float, float] = (0, 0.3),
    ) -> Normal:
        """Create a Normal from a simpler set of inputs.

        Returns
        -------
        Normal
        """
        return cls(
            n_features=n_features,
            n_layers=n_layers,
            coord_names=(coord_name,),
            param_names=ParamNames(("mixparam", (coord_name, ("mu", "sigma")))),  # type: ignore[arg-type] # noqa: E501
            coord_bounds=HashableMap({coord_name: coord_bounds}),  # type: ignore[arg-type] # noqa: E501
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    "mixparam": mixparam_bounds,
                    coord_name: HashableMap(
                        mu=mu_bounds,
                        sigma=sigma_bounds,
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
        lnp = xp.zeros_like(
            pars[
                "mixparam",
            ]
        )  # 100%
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
        return self.layers(args[0])
