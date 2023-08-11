"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from stream_ml.core._multi.bases import ModelsBase
from stream_ml.core.params._collection import ModelParameters
from stream_ml.core.params._values import Params, add_prefix, freeze_params, set_param
from stream_ml.core.setup_package import WEIGHT_NAME
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.cached_property import cached_property
from stream_ml.core.utils.funcs import get_prefixed_kwargs

if TYPE_CHECKING:
    from stream_ml.core._data import Data
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
    components : Mapping[str, Model], optional
        A mapping of the components of the model. The keys are the names of the
        component models, and the values are the models themselves. The names do
        not have to match the names on the model.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a composite model.

    priors : tuple[Prior, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.

    Notes
    -----
    The following fields on :class:`~stream_ml.core.ModelAPI` are properties here:

    - :attr:`~stream_ml.core.ModelBase.indep_coord_names`
    - :attr:`~stream_ml.core.ModelBase.coord_names`
    - :attr:`~stream_ml.core.ModelBase.coord_err_names`
    - :attr:`~stream_ml.core.ModelBase.coord_bounds`
    """

    @cached_property
    def indep_coord_names(self) -> tuple[str, ...]:  # type: ignore[override]
        """Independent coordinate names."""
        return tuple({n for m in self.components.values() for n in m.indep_coord_names})

    @cached_property
    def params(self) -> ModelParameters[Array]:  # type: ignore[override]
        cps: dict[str, ModelParameter[Array] | FrozenDict[str, ModelParameter[Array]]]
        cps = {
            f"{n}.{k}": v
            for n, m in self.components.items()
            for k, v in m.params.items()
        }
        return ModelParameters[Array](cps)

    @property
    def composite_params(self) -> ModelParameters[Array]:
        """Composite parameters."""
        return self.params

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
            {WEIGHT_NAME: extras[WEIGHT_NAME]}
            if extras is not None and WEIGHT_NAME in extras
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
                    m.unpack_params(marr, extras=mextras, freeze=False),
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
        self,
        mpars: Params[Array],
        /,
        data: Data[Array],
        *,
        where: Data[Array] | None = None,
        **kwargs: Array,
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

        where : Data[Array], optional keyword-only
            Where to evaluate the log-likelihood. If not provided, then the
            log-likelihood is evaluated at all data points.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        lnlik: Array = self.xp.zeros(())
        for name, m in self.components.items():
            lnlik = lnlik + m.ln_likelihood(
                mpars.get_prefixed(name),
                data,
                where=where,
                **get_prefixed_kwargs(name, kwargs),
            )
        return lnlik
