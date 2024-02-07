"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

from stream_mapper.core import NNField
from stream_mapper.core._api import SupportsXP
from stream_mapper.core._multi.bases import ModelsBase, SupportsComponentGetItem
from stream_mapper.core.params import (
    ModelParameter,
    ModelParameters,
    Params,
    add_prefix,
    freeze_params,
)
from stream_mapper.core.params._field import ModelParametersField
from stream_mapper.core.setup_package import BACKGROUND_KEY, WEIGHT_NAME
from stream_mapper.core.typing import Array, NNModel
from stream_mapper.core.utils.cached_property import cached_property
from stream_mapper.core.utils.funcs import get_prefixed_kwargs
from stream_mapper.core.utils.scale import DataScaler  # noqa: TCH001

if TYPE_CHECKING:
    from stream_mapper.core._data import Data
    from stream_mapper.core.typing import ParamNameAllOpts, ParamsLikeDict
    from stream_mapper.core.utils.frozen_dict import FrozenDict


class ComponentAllProbabilities(
    SupportsXP[Array], SupportsComponentGetItem[Array, NNModel]
):
    def component_ln_likelihood(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return (
            self[component].ln_likelihood(
                mpars.get_prefixed(component),
                data,
                where=where,
                **get_prefixed_kwargs(component, kwargs),
            )
            + mpars[(f"{component}.{WEIGHT_NAME}",)]
        )

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

    def component_ln_evidence(self, component: str, /, data: Data[Array]) -> Array:
        """Log-evidence of a component.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self[component].ln_evidence(data)

    def component_ln_posterior(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return (
            self[component].ln_posterior(
                mpars.get_prefixed(component),
                data,
                where=where,
                **get_prefixed_kwargs(component, kwargs),
            )
            + mpars[(f"{component}.{WEIGHT_NAME}",)]
        )

    # ----------------------------------------------------------------

    def component_ln_likelihood_tot(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(
            self.component_ln_likelihood(component, mpars, data, where=where, **kwargs)
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

    def component_ln_evidence_tot(self, component: str, /, data: Data[Array]) -> Array:
        """Sum of the component log-evidence.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self.xp.sum(self.component_ln_evidence(component, data))

    def component_ln_posterior_tot(
        self,
        component: str,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(
            self.component_ln_posterior(component, mpars, data, where=where, **kwargs)
        )

    # ----------------------------------------------------------------

    def component_likelihood(
        self,
        component: str,
        mpars: Params[Array],
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.exp(
            self.component_ln_likelihood(component, mpars, data, where=where, **kwargs)
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
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.exp(self.component_ln_prior(component, mpars, data))

    def component_evidence(self, component: str, data: Data[Array]) -> Array:
        """Evidence of a component.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self.xp.exp(self.component_ln_evidence(component, data))

    def component_posterior(
        self,
        component: str,
        mpars: Params[Array],
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.exp(
            self.component_ln_posterior(component, mpars, data, where=where, **kwargs)
        )

    # ----------------------------------------------------------------

    def component_likelihood_tot(
        self,
        component: str,
        mpars: Params[Array],
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(
            self.component_likelihood(component, mpars, data, where=where, **kwargs)
        )

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

    def component_evidence_tot(self, component: str, data: Data[Array]) -> Array:
        """Sum of the component evidence.

        Parameters
        ----------
        component : str, positional-only
            Component name.
        data : Data[Array], positional-only
            Data.

        Returns
        -------
        Array
        """
        return self.xp.sum(self.component_evidence(component, data))

    def component_posterior_tot(
        self,
        component: str,
        mpars: Params[Array],
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments, passed to the component's
            :meth:`~stream_mapper.core.ModelAPI.ln_likelihood``.

        Returns
        -------
        Array
        """
        return self.xp.sum(
            self.component_posterior(component, mpars, data, where=where, **kwargs)
        )


# ============================================================================


@dataclass(repr=False)
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
        a mixture model (see :class:`~stream_mapper.core.core.MixtureModel`).

    priors : tuple[Prior, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """

    net: NNField[NNModel, NNModel] = NNField()

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
            msg = f"the {BACKGROUND_KEY} model must be the last component."
            raise KeyError(msg)
        self._includes_bkg: bool = includes_bkg
        self._bkg_slc = slice(-1) if includes_bkg else slice(None)

    @cached_property
    def composite_params(self) -> ModelParameters[Array]:  # type: ignore[override]
        cps: dict[
            str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]
        ] = {}
        for n, m in self.components.items():
            cps[f"{n}.{WEIGHT_NAME}"] = self.params[f"{n}.{WEIGHT_NAME}"]
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
    ) -> ParamsLikeDict[Array]: ...

    @overload
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: Literal[True],
    ) -> Params[Array]: ...

    @overload
    def _unpack_params_from_arr(
        self,
        arr: Array,
        /,
        extras: dict[ParamNameAllOpts, Array] | None,
        *,
        freeze: bool,
    ) -> Params[Array] | ParamsLikeDict[Array]: ...

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
                ln_weight = arr[:, j]
            else:
                # The background is special, because it has a weight parameter
                # that is defined as 1 - the sum of the other weights.
                # So, we need to calculate the background weight from the
                # other weights. Note that the background weight can be included
                # in the parameter array, but it should not be determined by
                # any network output, rather just a placeholder.

                # The background weight is 1 - the other weights
                other_weights = self.xp.stack(
                    tuple(
                        cast("Array", pars[f"{k}.{WEIGHT_NAME}"])
                        for k in tuple(self.components.keys())[:-1]
                        # bkg is the last component
                    ),
                    1,
                )
                ln_weight = self.xp.log(
                    -self.xp.expm1(self.xp.special.logsumexp(other_weights, 1))
                )

            j += 1  # Increment the index (weight)

            # --- Parameters ---

            delta = len(m.params.flatkeys())

            # If there are no parameters, then just add the weight
            if delta == 0:
                # Add the component's parameters, prefixed with the component name
                pars[f"{n}.{WEIGHT_NAME}"] = ln_weight
                continue

            # Otherwise, get the relevant slice of the array
            marr = arr[:, slice(j, j + delta)]

            # Add the component's parameters, prefixed with the component name
            pars.update(
                add_prefix(
                    m.unpack_params(
                        marr,
                        extras=extras_ | {WEIGHT_NAME: ln_weight},  # pass the weight
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
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        (N,) Array
        """
        # Get the parameters for each model, stripping the model name,
        # and use that to evaluate the log likelihood for the model.
        lnliks = (  # (N,)
            self.component_ln_likelihood(name, mpars, data, where=where, **kwargs)
            for name in self.components
        )
        # Sum over the models, keeping the data dimension
        return self.xp.special.logsumexp(self.xp.vstack(tuple(lnliks)), 0)
