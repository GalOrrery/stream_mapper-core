"""Core feature."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

# LOCAL
from stream_ml.core.api import WEIGHT_NAME
from stream_ml.core.bases import ModelsBase
from stream_ml.core.params import ParamNames, Params
from stream_ml.core.typing import Array

if TYPE_CHECKING:
    # LOCAL
    from stream_ml.core.data import Data

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class IndependentModels(ModelsBase[Array]):
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

    def __post_init__(self) -> None:
        # Add the param_names  # TODO: make sure no duplicates
        # The first is the weight and it is shared across all components.
        self._param_names: ParamNames = ParamNames(
            (WEIGHT_NAME,)
            + tuple(
                (f"{c}.{p[0]}", p[1]) if isinstance(p, tuple) else f"{c}.{p}"
                for c, m in self.components.items()
                for p in m.param_names
                if p != WEIGHT_NAME
            ),
        )

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
        pars = dict[str, Array | Mapping[str, Array]]()

        # Do the weight first. This is shared across all components.
        # FIXME! should pass the weight even to the background model.
        pars[WEIGHT_NAME] = p_arr[:, :1]

        # Iterate through the components
        j = 1
        for n, m in self.components.items():  # iter thru models
            # Determine whether the model has parameters beyond the weight
            if len(m.param_names.flat) == 1:
                continue

            # Get weight and relevant parameters by index
            mp_arr = p_arr[:, [0] + list(range(j, j + len(m.param_names.flat) - 1))]

            # Skip empty (and incrementing the index)
            if mp_arr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars.update(m.unpack_params_from_arr(mp_arr).add_prefix(n + "."))

            # Increment the index
            j += len(m.param_names.flat) - 1

        return Params[Array](pars)

    def pack_params_to_arr(self, mpars: Params[Array], /) -> Array:  # noqa: D102
        raise NotImplementedError

    # ===============================================================
    # Statistics

    def ln_likelihood_arr(
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
        # TODO: this is a bit of a hack to start with 0. We should use a
        #       ``get_namespace`` method to get ``xp.zeros``.
        lnlik: Array = 0  # type: ignore[assignment]
        for name, m in self.components.items():
            # Get the kwargs for this component
            kwargs_ = self._get_prefixed_kwargs(name, kwargs)

            # Get the relevant parameters
            mpars_ = mpars.get_prefixed(name + ".")

            # Compute the log-likelihood
            mlnlik = m.ln_likelihood_arr(mpars_, data, **kwargs_)

            # Add to the total
            lnlik = lnlik + mlnlik

        return lnlik
