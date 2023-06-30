"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

from stream_ml.core import NNField
from stream_ml.core._api import SupportsXP
from stream_ml.core._multi.bases import ModelsBase, SupportsComponentGetItem
from stream_ml.core.params import (
    ModelParameter,
    ModelParameters,
    Params,
    add_prefix,
    freeze_params,
)
from stream_ml.core.params._field import ModelParametersField
from stream_ml.core.setup_package import BACKGROUND_KEY, WEIGHT_NAME
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.cached_property import cached_property
from stream_ml.core.utils.funcs import get_prefixed_kwargs
from stream_ml.core.utils.scale import DataScaler  # noqa: TCH001
from stream_ml.core.utils.sentinel import MISSING

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.typing import ParamNameAllOpts, ParamsLikeDict
    from stream_ml.core.utils.frozen_dict import FrozenDict


class ComponentAllProbabilities(
    SupportsXP[Array], SupportsComponentGetItem[Array, NNModel]
):
    def component_ln_likelihood(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        **kwargs: Array,
    ) -> Array:
        """Log-likelihood of a component, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self[component].ln_likelihood(
            mpars.get_prefixed(component),
            data,
            **get_prefixed_kwargs(component, kwargs),
        ) + self.xp.log(mpars[(f"{component}.weight",)])

    def component_ln_prior(
        self, component: str, mpars: Params[Array], /, data: Data[Array]
    ) -> Array:
        """Log-prior of a component.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self[component].ln_prior(mpars.get_prefixed(component), data)

    def component_ln_posterior(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        **kwargs: Array,
    ) -> Array:
        """Log-posterior of a component, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self[component].ln_posterior(
            mpars.get_prefixed(component),
            data,
            **get_prefixed_kwargs(component, kwargs),
        ) + self.xp.log(mpars[(f"{component}.weight",)])

    # ----------------------------------------------------------------

    def component_ln_likelihood_tot(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        **kwargs: Array,
    ) -> Array:
        """Sum of the component log-likelihood, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(
            self.component_ln_likelihood(component, mpars, data, **kwargs)
        )

    def component_ln_prior_tot(
        self, component: str, mpars: Params[Array], /, data: Data[Array]
    ) -> Array:
        """Sum of the component log-prior.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self.xp.sum(self.component_ln_prior(component, mpars, data))

    def component_ln_posterior_tot(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        **kwargs: Array,
    ) -> Array:
        """Sum of the component log-posterior, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(
            self.component_ln_posterior(component, mpars, data, **kwargs)
        )

    # ----------------------------------------------------------------

    def component_likelihood(
        self, component: str, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Likelihood of a component, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.exp(
            self.component_ln_likelihood(component, mpars, data, **kwargs)
        )

    def component_prior(
        self, component: str, mpars: Params[Array], data: Data[Array]
    ) -> Array:
        """Prior of a component.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.exp(self.component_ln_prior(component, mpars, data))

    def component_posterior(
        self, component: str, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Posterior of a component, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.exp(
            self.component_ln_posterior(component, mpars, data, **kwargs)
        )

    # ----------------------------------------------------------------

    def component_likelihood_tot(
        self, component: str, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Sum of the component likelihood, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(self.component_likelihood(component, mpars, data, **kwargs))

    def component_prior_tot(
        self, component: str, mpars: Params[Array], data: Data[Array]
    ) -> Array:
        """Sum of the component prior.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self.xp.sum(self.component_prior(component, mpars, data))

    def component_posterior_tot(
        self, component: str, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Sum of the component posterior, including the weight.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], positional-only
            Data.

        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_ml.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(self.component_posterior(component, mpars, data, **kwargs))


# ============================================================================


@dataclass
class MixtureModel(
    ModelsBase[Array, NNModel], ComponentAllProbabilities[Array, NNModel]
):
    """Full Model.

    Parameters
    ----------
    components : Mapping[str, Model], optional postional-only
        Mapping of Models. This allows for strict ordering of the Models and
        control over the type of the models attribute.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).

    priors : tuple[PriorBase, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """

    net: NNField[NNModel] = NNField(default=MISSING)

    _: KW_ONLY

    # Standardizer
    data_scaler: DataScaler[Array]

    # Coordinates, indpendent and dependent.
    indep_coord_names: tuple[str, ...] = ("phi1",)

    # Model Parameters, generally produced by the neural network.
    params: ModelParametersField[Array] = ModelParametersField[Array]()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Check if the model has a background component.
        # If it does, then it must be the last component.
        includes_bkg = BACKGROUND_KEY in self.components
        if includes_bkg and tuple(self.components.keys())[-1] != BACKGROUND_KEY:
            msg = "the background model must be the last component."
            raise KeyError(msg)
        self._includes_bkg: bool = includes_bkg
        self._bkg_slc = slice(-1) if includes_bkg else slice(None)

    @cached_property
    def composite_params(self) -> ModelParameters[Array]:  # type: ignore[override]
        cps: dict[
            str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]
        ] = {}
        for n, m in self.components.items():
            cps[f"{n}.weight"] = self.params[f"{n}.weight"]
            cps.update({f"{n}.{k}": v for k, v in m.params.items()})
        return ModelParameters[Array](cps)

    # ===============================================================

    @overload
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[False],
    ) -> ParamsLikeDict[Array]:
        ...

    @overload
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[True],
    ) -> Params[Array]:
        ...

    @overload
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        ...

    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]:
        """Unpack parameters into a dictionary.

        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        arr : Array
            Parameter array.
        extras : dict[ParamNameAllOpts, Array] | None, keyword-only
            Extra arrays to add.
        freeze : bool, optional keyword-only
            Whether to freeze the parameters. Default is `True`.

        Returns
        -------
        Params[Array]
        """
        extras_: dict[ParamNameAllOpts, Array] = {} if extras is None else extras

        # Unpack the parameters
        pars: ParamsLikeDict[Array] = {}
        j: int = 0
        for n, m in self.components.items():  # iter thru models
            # Weight
            if n != BACKGROUND_KEY:
                weight = arr[:, j]
            else:
                # The background is special, because it has a weight parameter
                # that is defined as 1 - the sum of the other weights.
                # So, we need to calculate the background weight from the
                # other weights. Note that the background weight can be included
                # in the parameter array, but it should not be determined by
                # any network output, rather just a placeholder.

                # The background weight is 1 - the other weights
                weight = 1 - sum(
                    (
                        cast("Array", pars[f"{k}.weight"])
                        for k in tuple(self.components.keys())[:-1]
                        # skipping the background, which is the last component
                    ),
                    start=self.xp.zeros(len(arr), dtype=arr.dtype),
                )

            j += 1  # Increment the index (weight)

            # --- Parameters ---

            delta = len(m.params.flatkeys())

            # If there are no parameters, then just add the weight
            if delta == 0:
                # Add the component's parameters, prefixed with the component name
                pars[f"{n}.weight"] = weight
                continue

            # Otherwise, get the relevant slice of the array
            marr = arr[:, slice(j, j + delta)]

            # Add the component's parameters, prefixed with the component name
            pars.update(
                add_prefix(
                    m.unpack_params(
                        marr,
                        extras=extras_ | {WEIGHT_NAME: weight},  # pass the weight
                        freeze=False,
                    ),
                    n + ".",
                )
            )
            j += delta  # Increment the index (parameters)

        # Allow for conversation between components
        for hook in self.unpack_params_hooks:
            pars = hook(pars)

        return freeze_params(pars) if freeze else pars

    # ===============================================================

    def ln_likelihood(
        self, mpars: Params[Array], /, data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods, including the
        weights.

        Parameters
        ----------
        mpars : (N,) Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : (N, F) Data[Array]
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        (N,) Array
        """
        # Get the parameters for each model, stripping the model name,
        # and use that to evaluate the log likelihood for the model.
        lnliks = (
            self.component_ln_likelihood(name, mpars, data, **kwargs)  # (N,)
            for name in self.components
        )
        # Sum over the models, keeping the data dimension
        return self.xp.special.logsumexp(self.xp.vstack(tuple(lnliks)), 0)
