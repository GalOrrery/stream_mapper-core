"""Core feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stream_ml.core.multi._bases import ModelsBase
from stream_ml.core.params import ParamBounds, ParamNames, Params, ParamScalers
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.frozen_dict import FrozenDictField
from stream_ml.core.utils.funcs import get_prefixed_kwargs

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

        super().__post_init__()

        if self.has_weight.keys() != self.components.keys():
            msg = "has_weight must match components"
            raise ValueError(msg)
        elif not any(self.has_weight.values()):
            msg = "there must be at least one weight"
            raise ValueError(msg)

        # Add the param_names
        # The first is the weight and it is shared across all components.
        self._param_names: ParamNames = ParamNames(
            (WEIGHT_NAME, *tuple(self._param_names))
        )

        # Add the param_bounds  # TODO! not update internal to ParamBounds.
        cps: ParamBounds[Array] = ParamBounds()
        for n, m in self.components.items():
            cps._dict.update({f"{n}.{k}": v for k, v in m.param_bounds.items()})
        self._param_bounds = cps

        # Add the param_scalers  # TODO! not update internal to ParamScalers.
        pss: ParamScalers[Array] = ParamScalers()
        for n, m in self.components.items():
            pss._dict.update({f"{n}.{k}": v for k, v in m.param_scalers.items()})
        self._param_scalers = pss

    @property  # type: ignore[override]
    def param_names(self) -> ParamNames:
        """Parameter names."""
        return self._param_names

    @param_names.setter  # hack to match the Protocol
    def param_names(self, value: Any) -> None:
        """Set the parameter names."""
        msg = "cannot set param_names"
        raise AttributeError(msg)

    @property  # type: ignore[override]
    def param_bounds(self) -> ParamBounds[Array]:
        """Coordinate names."""
        return self._param_bounds

    @param_bounds.setter  # hack to match the Protocol
    def param_bounds(self, value: Any) -> None:
        """Set the parameter bounds."""
        msg = "cannot set param_bounds"
        raise AttributeError(msg)

    @property  # type: ignore[override]
    def param_scalers(self) -> ParamScalers[Array]:
        """Parameter scalers."""
        return self._param_scalers

    @param_scalers.setter  # hack to match the Protocol
    def param_scalers(self, value: Any) -> None:
        """Set the parameter scalers."""
        msg = "cannot set param_scalers"
        raise AttributeError(msg)

    # ===============================================================

    def unpack_params_from_arr(self, arr: Array) -> Params[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        arr : Array
            Parameter array.

        Returns
        -------
        Params[Array]
        """
        # Unpack the parameters
        pars: dict[str, Array | Mapping[str, Array]] = {}

        # Add the weight.  # TODO! more general index
        pars[WEIGHT_NAME] = arr[:, 0:1]

        # Iterate through the components
        j: int = 1
        for n, m in self.components.items():  # iter thru models
            # Determine whether the model has parameters beyond the weight
            if len(m.param_names.flat) == 0:
                continue

            # number of parameters, minus the weight
            delta = len(m.param_names.flat) - (1 if self.has_weight[n] else 0)

            # Get weight and relevant parameters by index
            marr = arr[:, [0, *list(range(j, j + delta))]]

            # Skip empty (and incrementing the index)
            if marr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars.update(m.unpack_params_from_arr(marr).add_prefix(n + "."))

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
                mpars.get_prefixed(name), data, **get_prefixed_kwargs(name, kwargs)
            )
        return lnlik
