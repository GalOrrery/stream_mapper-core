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
from stream_ml.core.utils.hashdict import FrozenDict, FrozenDictField
from stream_ml.core.utils.params import (
    ParamBounds,
    ParamBoundsField,
    ParamNames,
    Params,
)
from stream_ml.jax.stream.base import StreamModel
from stream_ml.jax.utils import within_bounds
from stream_ml.jax.utils.sigmoid import ColumnarScaledSigmoid
from stream_ml.jax.utils.tanh import Tanh

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.jax._typing import Array, DataT

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

    coord_bounds: FrozenDictField[str, tuple[float, float]] = FrozenDictField()
    param_bounds: ParamBoundsField = ParamBoundsField(ParamBounds())

    def __post_init__(self) -> None:
        super().__post_init__()

        # Validate the coord_names
        if len(self.coord_names) != 1:
            raise ValueError("Only one coordinate is supported, e.g ('phi2',).")

        # Validate the param_names
        if self.param_names != ("mixparam", (self.coord_names[0], ("mu", "sigma"))):
            raise ValueError(
                "param_names must be ('sigma', (<coordinate>, ('mu', 'sigma')))."
            )

        # Validate the param_bounds
        for pn in self.param_names:
            # "in X" ignores __contains__ & __getitem__ signatures
            if not self.param_bounds.__contains__(pn):
                raise ValueError(f"param_bounds must contain {pn}.")
            # TODO: recursively check for all sub-parameters
        # [("mixparam", (0.0, 1.0)), ("mu", (-xp.inf, xp.inf)), ("sigma", (0.0, 0.3))]

    def setup(self) -> None:
        """Setup."""
        cn = self.coord_names[0]

        self.layers = nn.Sequential(
            nn.Dense(self.n_features),
            Tanh(),
            *functools.reduce(
                operator.add,
                ((nn.Dense(self.n_features), Tanh()) for _ in range(self.n_layers - 2)),
            ),
            nn.Dense(3),
            ColumnarScaledSigmoid(
                (0, 2),
                (
                    self.param_bounds[("mixparam",)],
                    self.param_bounds[cn, "sigma"],
                ),
            ),
            name=self.name,
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
        mu_bounds: tuple[float, float] = (-xp.inf, xp.inf),
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
            coord_bounds=FrozenDict({coord_name: coord_bounds}),  # type: ignore[arg-type] # noqa: E501
            param_bounds=FrozenDict(  # type: ignore[arg-type]
                mixparam=mixparam_bounds,
                coord_name=FrozenDict(
                    mu=mu_bounds,
                    sigma=sigma_bounds,
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
        pars : Params[Array]
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
        return xp.log(xp.clip(pars[("mixparam",)], 0)) + norm.logpdf(
            data[c], pars[(c, "mu")], xp.clip(pars[(c, "sigma")], min=1e-10)
        )

    def ln_prior_arr(self, pars: Params[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.

        Returns
        -------
        Array
        """
        lnp = xp.zeros_like(pars[("mixparam",)])  # 100%
        # Bounds
        for name, bounds in self.param_bounds.items():
            lnp[~within_bounds(pars[name], *bounds)] = -xp.inf
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
        pred = self.layers(args[0])
        return pred
