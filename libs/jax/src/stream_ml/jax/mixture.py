"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# THIRD-PARTY
import flax.linen as nn
import jax.numpy as xp
from jax.scipy.special import logsumexp

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.mixture import MixtureModelBase
from stream_ml.core.params import Params
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.jax.base import Model
from stream_ml.jax.typing import Array

__all__: list[str] = []


@dataclass()
class MixtureModel(nn.Module, MixtureModelBase[Array], Model):  # type: ignore[misc]
    """Full Model.

    Parameters
    ----------
    models : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    **more_models : Model
        Additional Models.
    """

    # Need to override this because of the type hinting
    components: FrozenDictField[str, Model] = FrozenDictField()  # type: ignore[assignment]  # noqa: E501

    def setup(self) -> None:
        """Setup ML."""
        # TODO!

    def pack_params_to_arr(self, pars: Params[Array]) -> Array:
        """Pack parameters into an array.

        Parameters
        ----------
        pars : Params
            Parameter dictionary.

        Returns
        -------
        Array
        """
        return Model.pack_params_to_arr(self, pars)

    # ===============================================================
    # Statistics

    def ln_likelihood_arr(
        self, pars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # Get the parameters for each model, stripping the model name,
        # and use that to evaluate the log likelihood for the model.
        liks = tuple(
            model.ln_likelihood_arr(pars.get_prefixed(name), data, **kwargs)
            for name, model in self.components.items()
        )
        # Sum over the models, keeping the data dimension
        return logsumexp(xp.hstack(liks), axis=1)[:, None]

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data: Data[Array]
            Data.

        Returns
        -------
        Array
        """
        # Get the parameters for each model, stripping the model name,
        # and use that to evaluate the log prior for the model.
        lps = tuple(
            model.ln_prior_arr(pars.get_prefixed(name), data)
            for name, model in self.components.items()
        )
        lp = xp.hstack(lps).sum(dim=1)[:, None]

        # Plugin for priors
        for prior in self.priors:
            lp += prior.logpdf(pars, data, self, lp)

        # Sum over the priors
        return lp

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
        result = xp.concatenate(
            [model(*args) for model in self.components.values()], axis=1
        )

        # Call the prior to limite the range of the parameters
        # TODO: full data, not args[0]
        for prior in self.priors:
            result = prior(result, args[0], self)

        return result
