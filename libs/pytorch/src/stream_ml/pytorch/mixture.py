"""Core feature."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp
import torch.nn as nn

# LOCAL
from stream_ml.core.mixture import MixtureModelBase
from stream_ml.core.utils.hashdict import FrozenDictField
from stream_ml.core.utils.params import Params
from stream_ml.pytorch._typing import Array
from stream_ml.pytorch.base import Model

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import DataT

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
        self, pars: Params[Array], data: DataT, *args: Array
    ) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

        Parameters
        ----------
        pars : Params
            Parameters.
        data : DataT
            Data.
        args : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        # (n_models, n_dat, 1)
        liks = []
        for name, model in self.items():
            # Get the parameters for this model, stripping the model name
            mps = pars.get_prefixed(name)
            # Add the likelihood
            lik = model.ln_likelihood_arr(mps, data, *args)
            liks.append(lik)

        # Sum over the models, keeping the data dimension
        return xp.logsumexp(xp.hstack(liks), dim=1)[:, None]

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
        ps = []
        for name, model in self.components.items():
            # Get the parameters for this model, stripping the model name
            mps = pars.get_prefixed(name)
            # Add the prior
            ps.append(model.ln_prior_arr(mps))

        # Plugin for priors
        for hook in self._hook_prior.values():
            ps.append(hook(pars))

        # Sum over the priors
        return xp.hstack(ps).sum(dim=1)[:, None]

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
        return xp.concat([model(*args) for model in self.components.values()], dim=1)
