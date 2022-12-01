"""Core feature."""

from __future__ import annotations

# STDLIB
import abc
from collections.abc import ItemsView, Iterator, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Callable

# THIRD-PARTY
import numpy as np

# LOCAL
from stream_ml.core.base import ModelBase

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core._typing import ArrayT, DataT, ParsT

__all__: list[str] = []


@dataclass
class MixtureModelBase(ModelBase[ArrayT], Mapping[str, ModelBase[ArrayT]]):
    """Full Model.

    Parameters
    ----------
    models : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.
    **more_models : Model
        Additional Models.
    """

    _models: Mapping[str, ModelBase[ArrayT]] = field(default_factory=dict)
    _: KW_ONLY
    tied_params: dict[str, Callable[[ParsT[ArrayT]], ArrayT]] = field(
        default_factory=dict
    )
    hook_prior: Mapping[str, Callable[[ParsT[ArrayT]], ArrayT]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        super().__init__()

        try:
            super().__post_init__()  # type: ignore[misc]
        except AttributeError:
            pass

    @property
    def models(self) -> ItemsView[str, ModelBase[ArrayT]]:
        """Models (view)."""
        return self._models.items()

    @property
    def param_names(self) -> dict[str, int]:
        """Parameter names, flattening over the models."""
        return {
            f"{n}_{p}": v
            for n, m in self._models.items()
            for p, v in m.param_names.items()
        }

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> ModelBase[ArrayT]:
        return self._models[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def __hash__(self) -> int:
        return hash(tuple(self.keys()))

    # ===============================================================

    def unpack_pars(self, p_arr: ArrayT) -> ParsT[ArrayT]:
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
            param_inds = np.array(
                tuple(
                    i
                    for i, p in enumerate(m.param_names)
                    if f"{n}_{p}" not in self.tied_params
                )
            )
            mp_arr = p_arr[:, j + param_inds]

            # Skip empty
            if mp_arr.shape[1] == 0:
                continue

            mp_dict = m.unpack_pars(mp_arr)
            for k, p in mp_dict.items():
                p_dict[f"{n}_{k}"] = p

        # Add the dependent parameters
        for name, tie in self.tied_params.items():
            p_dict[name] = tie(p_dict)

        return p_dict

    # ===============================================================
    # Statistics

    @abc.abstractmethod
    def ln_likelihood(
        self, pars: ParsT[ArrayT], data: DataT[ArrayT], *args: ArrayT
    ) -> ArrayT:
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
        raise NotImplementedError

    @abc.abstractmethod
    def ln_prior(self, pars: ParsT[ArrayT]) -> ArrayT:
        """Log prior.

        Parameters
        ----------
        pars : ParsT
            Parameters.

        Returns
        -------
        Array
        """
        raise NotImplementedError
