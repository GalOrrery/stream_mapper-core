"""Core feature."""

from __future__ import annotations

# STDLIB
import functools
import operator
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import flax.linen as nn
import jax.numpy as xp
from jax.scipy.stats import norm

# LOCAL
from stream_ml.core._typing import BoundsT
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBounds, ParamBoundsField, ParamNames, Params
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.utils.hashdict import FrozenDict, FrozenDictField
from stream_ml.jax.prior.bounds import PriorBounds
from stream_ml.jax.stream.base import StreamModel
from stream_ml.jax.utils.tanh import Tanh

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.jax._typing import Array

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
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField()
    param_names: ParamNamesField = ParamNamesField(("weight", (..., ("mu", "sigma"))))
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](ParamBounds())

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

        cn = self.coord_names[0]

        # Validate the param_names
        if self.param_names != ("weight", (cn, ("mu", "sigma"))):
            msg = (
                f"param_names must be ('weight', ({cn}, ('mu', 'sigma'))),"
                f" gott {self.param_names}"
            )
            raise ValueError(msg)

        # Validate the param_bounds
        for pn in self.param_names:
            # "in X" ignores __contains__ & __getitem__ signatures
            if not self.param_bounds.__contains__(pn):
                msg = f"param_bounds must contain {pn}."
                raise ValueError(msg)
            # TODO: recursively check for all sub-parameters

    def setup(self) -> None:
        """Setup."""
        self.layers = nn.Sequential(
            nn.Dense(self.n_features),
            Tanh(),
            *functools.reduce(
                operator.add,
                ((nn.Dense(self.n_features), Tanh()) for _ in range(self.n_layers - 2)),
            ),
            nn.Dense(3),
            name=self.name,
        )

    @classmethod
    def from_simpler_inputs(
        cls,
        n_features: int = 50,
        n_layers: int = 3,
        *,
        coord_name: str,
        coord_bounds: PriorBounds | BoundsT,
        weight_bounds: PriorBounds | BoundsT = (0, 1),
        mu_bounds: PriorBounds | BoundsT = (-xp.inf, xp.inf),
        sigma_bounds: PriorBounds | BoundsT = (0, 0.3),
    ) -> Normal:
        """Create a Normal from a simpler set of inputs.

        Parameters
        ----------
        n_features : int, optional
            Number of features, by default 50.
        n_layers : int, optional
            Number of hidden layers, by default 3.

        coord_name : str, keyword-only
            Coordinate name.
        coord_bounds : BoundsT, keyword-only
            Coordinate bounds.
        weight_bounds : PriorBounds | BoundsT, keyword-only
            Mixparam bounds, by default (0, 1).
        mu_bounds : PriorBounds | BoundsT, keyword-only
            Mu bounds, by default (-xp.inf, xp.inf).
        sigma_bounds : PriorBounds | BoundsT, keyword-only
            Sigma bounds, by default (0, 0.3).

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
        min_ = xp.finfo(pars[("weight",)].dtype).eps  # TOOD: or tiny?

        return xp.log(xp.clip(pars[("weight",)], min_)) + norm.logpdf(
            data[c], pars[(c, "mu")], xp.clip(pars[(c, "sigma")], a_min=min_)
        )

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data (phi1, phi2).

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("weight",)])  # 100%

        # Bounds
        lnp += self._ln_prior_coord_bnds(pars, data)
        for bounds in self.param_bounds.flatvalues():
            lnp += bounds.logpdf(pars, data, self, lnp)
        return lnp

    # ========================================================================
    # ML

    def __call__(self, *args: Array) -> Array:
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
        return self._forward_prior(self.layers(args[0]), args[0])
