"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import ItemsView, Iterator, Mapping
from typing import TYPE_CHECKING, Callable

# THIRD-PARTY
import numpy as np
import torch as xp

# LOCAL
from stream_ml.base import Model

if TYPE_CHECKING:
    # LOCAL
    from stream_ml._typing import Array, DataT, ParsT

__all__: list[str] = []


class CompositeModel(Model, Mapping[str, Model]):
    """Full Model.

    Parameters
    ----------
    models : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    **more_models : Model
        Additional Models.
    """

    def __init__(
        self,
        models: Mapping[str, Model] | list[tuple[str, Model]] | None = None,
        /,
        tied_params: dict[str, Callable[[ParsT], Array]] | None = None,  # noqa: N805
        hook_prior: Mapping[str, Callable[[ParsT], Array]] | None = None,
    ) -> None:
        super().__init__()

        self._models: Mapping[str, Model]
        if models is None:
            self._models = {}
        elif isinstance(models, Mapping):
            self._models = models
        else:
            self._models = dict(models)

        self._tied = tied_params if tied_params is not None else {}
        self._hook_prior: Mapping[str, Callable[[ParsT], Array]]
        if hook_prior is None:
            self._hook_prior = {}
        else:
            self._hook_prior = hook_prior

        # NOTE: don't need this in JAX
        for name, model in self._models.items():
            self.add_module(name=name, module=model)

    @property
    def models(self) -> ItemsView[str, Model]:
        """Models (view)."""
        return self._models.items()

    @property
    def tied_params(self) -> ItemsView[str, Callable[[ParsT], Array]]:
        """Tied parameters (view)."""
        return self._tied.items()

    @property
    def param_names(self) -> dict[str, int]:  # type: ignore[override]
        """Parameter names, flattening over the models."""
        return {f"{n}_{p}": v for n, m in self._models.items() for p, v in m.param_names.items()}

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> Model:
        return self._models[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def __hash__(self) -> int:
        return hash(tuple(self.keys()))

    # ===============================================================

    def unpack_pars(self, p_arr: Array) -> ParsT:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        p_arr : Array
            Parameter array.

        Returns
        -------
        ParsT
        """
        # Unpack the parameters
        p_dict = {}
        for j, (n, m) in enumerate(self._models.items()):  # iter thru models
            # Get relevant parameters by index
            param_inds = np.array(tuple(i for i, p in enumerate(m.param_names) if f"{n}_{p}" not in self._tied))
            mp_arr = p_arr[:, j + param_inds]

            # Skip empty
            if mp_arr.shape[1] == 0:
                continue

            mp_dict = m.unpack_pars(mp_arr)
            for k, p in mp_dict.items():
                p_dict[f"{n}_{k}"] = p

        # Add the dependent parameters
        for name, tie in self._tied.items():
            p_dict[name] = tie(p_dict)

        return p_dict

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
        liks = xp.stack([xp.exp(model.ln_likelihood(pars, data, *args)) for model in self._models.values()])
        lik = liks.sum(dim=0)  # (n_dat, 1)
        return xp.log(lik)

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
        return xp.stack([model.ln_prior(pars) for model in self._models.values()]).sum()

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
