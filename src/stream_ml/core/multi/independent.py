"""Core feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stream_ml.core.multi.bases import ModelsBase
from stream_ml.core.params import ParamNames, Params
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.frozen_dict import FrozenDictField

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stream_ml.core.data import Data

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class IndependentModels(ModelsBase[Array, NNModel]):
    """Composite of a few models that acts like one model.

    This is different from a mixture model in that the components are not
    separate, but are instead combined into a single model. Practically, this
    means:

    - All the components have the same weight.
    - The log-likelihoood of the composite model is the sum of the
      log-likelihooods of the components, not the log-sum-exp.

    Parameters
    ----------
    components : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a composite model.

    priors : tuple[PriorBase, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """

    has_weight: FrozenDictField[str, bool] = FrozenDictField()

    def __post_init__(self) -> None:
        self._mypyc_init_descriptor()  # TODO: Remove this when mypyc is fixed.

        # Add the param_names  # TODO: make sure no duplicates
        # The first is the weight and it is shared across all components.
        self._param_names: ParamNames = ParamNames(
            (
                WEIGHT_NAME,
                *tuple(
                    (f"{c}.{p[0]}", p[1]) if isinstance(p, tuple) else f"{c}.{p}"
                    for c, m in self.components.items()
                    for p in m.param_names
                    if p != WEIGHT_NAME
                ),
            ),
        )

        if self.has_weight.keys() != self.components.keys():
            msg = "has_weight must match components"
            raise ValueError(msg)
        elif not any(self.has_weight.values()):
            msg = "there must be at least one weight"
            raise ValueError(msg)

        super().__post_init__()

    # ===============================================================

    def unpack_params_from_arr(self, p_arr: Array) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        p_arr : Array
            Parameter array.

        Returns
        -------
        Params[Array]
        """
        # Unpack the parameters
        pars: dict[str, Array | Mapping[str, Array]] = {}

        # Add the weight.  # TODO! more general index
        pars[WEIGHT_NAME] = p_arr[:, 0:1]

        # Iterate through the components
        j: int = 1
        for n, m in self.components.items():  # iter thru models
            # Determine whether the model has parameters beyond the weight
            if len(m.param_names.flat) == 0:
                continue

            # number of parameters, minus the weight
            delta = len(m.param_names.flat) - (1 if self.has_weight[n] else 0)

            # Get weight and relevant parameters by index
            mp_arr = p_arr[:, [0, *list(range(j, j + delta))]]

            # Skip empty (and incrementing the index)
            if mp_arr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars.update(m.unpack_params_from_arr(mp_arr).add_prefix(n + "."))

            # Increment the index
            j += delta

        for hook in self.unpack_params_hooks:
            pars = hook(pars)

        return Params(pars)

    # ===============================================================
    # Statistics

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood.

        Just the summation of the individual log-likelihoods.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        lnlik: Array = self.xp.zeros(())
        for name, m in self.components.items():
            lnlik = lnlik + m.ln_likelihood(
                mpars.get_prefixed(name + "."),
                data,
                **self._get_prefixed_kwargs(name, kwargs),
            )

        return lnlik / len(self.components)
