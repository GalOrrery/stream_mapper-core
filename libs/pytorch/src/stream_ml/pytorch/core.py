"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Callable

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.core.core import MixtureModelBase
from stream_ml.core.utils import get_params_for_model
from stream_ml.pytorch.base import Model

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.pytorch._typing import Array, DataT, ParsT

__all__: list[str] = []


@dataclass
class MixtureModel(MixtureModelBase[Array], Model):
    """Full Model.

    Parameters
    ----------
    models : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    **more_models : Model
        Additional Models.
    """

    _models: Mapping[str, Model] = field(default_factory=dict)
    _: KW_ONLY
    tied_params: dict[str, Callable[[ParsT], Array]] = field(default_factory=dict)
    hook_prior: Mapping[str, Callable[[ParsT], Array]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()

        # NOTE: don't need this in JAX
        for name, model in self._models.items():
            self.add_module(name=name, module=model)

    # ===============================================================
    # Statistics

    def ln_likelihood(self, pars: ParsT, data: DataT, *args: Array) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

        Parameters
        ----------
        pars : ParsT
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
            mps = get_params_for_model(name, pars)
            # Add the likelihood
            lik = model.ln_likelihood(mps, data, *args)
            liks.append(lik)

        # Sum over the models, keeping the data dimension
        return xp.logsumexp(xp.hstack(liks), dim=1)[:, None]

    def ln_prior(self, pars: ParsT) -> Array:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Array
        """
        ps = []
        for name, model in self._models.items():
            # Get the parameters for this model, stripping the model name
            mps = get_params_for_model(name, pars)
            # Add the prior
            ps.append(model.ln_prior(mps))

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
        return xp.concat([model(*args) for model in self._models.values()], dim=1)
