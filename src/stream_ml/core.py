"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

# THIRD-PARTY
import torch as xp

# LOCAL
from stream_ml.base import Model

if TYPE_CHECKING:
    # THIRD-PARTY
    from torch import Tensor

    # LOCAL
    from stream_ml._typing import DataT, ParsT

__all__: list[str] = []


class CompositeModel(Model, Mapping[str, Model]):
    """Full Model.

    Parameters
    ----------
    models : Model
        Models.
    """

    def __init__(self, **models: Model) -> None:
        self.models = models

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> Model:
        return self.models[key]

    def __setitem__(self, key: str, value: Model) -> None:
        self.models[key] = value

    def __iter__(self) -> Iterator[str]:
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    # ===============================================================

    def ln_likelihood(self, pars: ParsT, data: DataT, *args: Tensor) -> Tensor:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

        Parameters
        ----------
        pars : ParsT
            Parameters.
        data : DataT
            Data.
        args : Tensor
            Additional arguments.

        Returns
        -------
        Tensor
        """
        # (n_models, n_dat, 1)
        liks = xp.stack([xp.exp(model.ln_likelihood(pars, data, *args)) for model in self.models.values()])
        lik = liks.sum(dim=0)  # (n_dat, 1)
        return xp.log(lik)

    def ln_prior(self, pars: ParsT) -> Tensor:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Tensor
        """
        return xp.stack([model.ln_prior(pars) for model in self.models.values()]).sum()
