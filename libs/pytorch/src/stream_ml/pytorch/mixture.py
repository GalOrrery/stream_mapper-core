"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# THIRD-PARTY
import torch as xp
from torch import nn

# LOCAL
from stream_ml.core.data import Data
from stream_ml.core.mixture import MixtureModelBase
from stream_ml.core.params import Params
from stream_ml.core.utils.hashdict import FrozenDictField
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.base import Model

__all__: list[str] = []


@dataclass(unsafe_hash=True)
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
    components: FrozenDictField[str, Model] = FrozenDictField[str, Model]()  # type: ignore[assignment]  # noqa: E501

    def __post_init__(self) -> None:
        super().__post_init__()

        # Register the models with pytorch.
        for name, model in self.components.items():
            self.add_module(name=name, module=model)

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
        pars : Params
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
            model.ln_likelihood_arr(
                pars.get_prefixed(name), data, **self._get_prefixed_kwargs(name, kwargs)
            )
            for name, model in self.components.items()
        )
        # Sum over the models, keeping the data dimension
        return xp.logsumexp(xp.hstack(liks), dim=1, keepdim=True)

    def ln_prior_arr(self, pars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : Params[Array]
            Parameters.
        data : Data[Array]
            Data (phi1).

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

    def forward(self, data: Data[Array], /) -> Array:
        """Forward pass.

        Parameters
        ----------
        data : Data
            Input.

        Returns
        -------
        Array
            fraction, mean, sigma
        """
        nn = xp.concat([model(data) for model in self.components.values()], dim=1)

        # Call the prior to limit the range of the parameters
        # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            nn = prior(nn, data, self)

        return nn
