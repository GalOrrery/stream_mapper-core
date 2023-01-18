"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import KW_ONLY, dataclass
from typing import Any

# THIRD-PARTY
import flax.linen as nn
import jax.numpy as xp
from jax.scipy.stats import norm

# LOCAL
from stream_ml.core.api import WEIGHT_NAME
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBounds, ParamBoundsField, ParamNames, Params
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.typing import BoundsT
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField
from stream_ml.jax.prior.bounds import PriorBounds
from stream_ml.jax.stream.base import StreamModel
from stream_ml.jax.typing import Array
from stream_ml.jax.utils.tanh import Tanh

__all__: list[str] = []


@dataclass
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
    param_names: ParamNamesField = ParamNamesField(
        (WEIGHT_NAME, (..., ("mu", "sigma")))
    )
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](ParamBounds())

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            msg = "Only one coordinate is supported, e.g ('phi2',)"
            raise ValueError(msg)

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
            param_names=ParamNames((WEIGHT_NAME, (coord_name, ("mu", "sigma")))),  # type: ignore[arg-type] # noqa: E501
            coord_bounds=FrozenDict({coord_name: coord_bounds}),  # type: ignore[arg-type] # noqa: E501
            param_bounds=ParamBounds(  # type: ignore[arg-type]
                {
                    WEIGHT_NAME: cls._make_bounds(weight_bounds, (WEIGHT_NAME,)),
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
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log-likelihood of the stream.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        c = self.coord_names[0]
        min_ = xp.finfo(mpars[(WEIGHT_NAME,)].dtype).eps  # TOOD: or tiny?

        return xp.log(xp.clip(mpars[(WEIGHT_NAME,)], min_)) + norm.logpdf(
            data[c], mpars[(c, "mu")], xp.clip(mpars[(c, "sigma")], a_min=min_)
        )

    def ln_prior_arr(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(mpars[(WEIGHT_NAME,)])  # 100%

        # Bounds
        lnp += self._ln_prior_coord_bnds(mpars, data)
        for bounds in self.param_bounds.flatvalues():
            lnp += bounds.logpdf(mpars, data, self, lnp)

        # TODO: use super().ln_prior_arr(mpars, data, current_lnp) once
        #       the last argument is added to the signature.
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp)

        return lnp

    # ========================================================================
    # ML

    @nn.compact  # type: ignore[misc]
    def __call__(self, *args: Array, **kwargs: Any) -> Array:
        """Forward pass.

        Parameters
        ----------
        *args : Array
            Input. Only uses the first argument.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        x = nn.Dense(self.n_features)(args[0])
        x = Tanh()(x)
        for _ in range(self.n_layers - 2):
            x = nn.Dense(self.n_features)(x)
            x = Tanh()(x)
        x = nn.Dense(3)(x)

        return self._forward_prior(x, x)
