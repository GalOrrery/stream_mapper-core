"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from stream_ml.core._multi.bases import ModelsBase
from stream_ml.core.params._collection import ModelParameters
from stream_ml.core.params._values import Params, add_prefix, freeze_params, set_param
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.cached_property import cached_property
from stream_ml.core.utils.funcs import get_prefixed_kwargs

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params._core import ModelParameter
    from stream_ml.core.typing import ParamNameAllOpts, ParamsLikeDict
    from stream_ml.core.utils.frozen_dict import FrozenDict


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

    @cached_property
    def params(self) -> ModelParameters[Array]:  # type: ignore[override]
        cps: dict[
            str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]
        ] = {}
        for n, m in self.components.items():
            cps.update({f"{n}.{k}": v for k, v in m.params.items()})
        return ModelParameters[Array](cps)

    # ===============================================================

    @overload
    def unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[False],
    ) -> ParamsLikeDict[Array]:
        ...

    @overload
    def unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[True] = ...,
    ) -> Params[Array]:
        ...

    @overload
    def unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        ...

    def unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None = None,
        *,
        freeze: bool = True,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        arr : Array
            Parameter array.
        extras : dict[ParamNameAllOpts, Array] | None, optional
            Extra parameters to add.
        freeze : bool, optional keyword-only
            Whether to freeze the parameters. Default is `True`.

        Returns
        -------
        Params[Array]
        """
        # Unpack the parameters
        pars: ParamsLikeDict[Array] = {}

        mextras: dict[ParamNameAllOpts, Array] | None = (
            {"weight": extras["weight"]}
            if extras is not None and "weight" in extras
            else None
        )

        # Iterate through the components
        j: int = 0
        for n, m in self.components.items():  # iter thru models
            # number of parameters
            delta = len(m.params.flatkeys())

            # Determine whether the model has parameters
            if delta == 0:
                continue

            # Get weight and relevant parameters by index
            marr = arr[:, list(range(j, j + delta))]

            # Skip empty (and incrementing the index)
            if marr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars.update(
                add_prefix(
                    m.unpack_params_from_arr(marr, extras=mextras, freeze=False),
                    n + ".",
                )
            )

            # Increment the index
            j += delta

        # Add the extras
        for k, v in (extras or {}).items():
            set_param(pars, k, v)

        # Apply the unpack_params_hooks
        for hook in self.unpack_params_hooks:
            pars = hook(pars)

        return freeze_params(pars) if freeze else pars

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
