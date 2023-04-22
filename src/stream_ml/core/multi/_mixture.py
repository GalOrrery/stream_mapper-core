"""Core feature."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, cast

from stream_ml.core import NNField
from stream_ml.core.multi._bases import ModelsBase
from stream_ml.core.params import ParamBounds, ParamNames, Params
from stream_ml.core.params.bounds import MixtureParamBoundsField
from stream_ml.core.params.scales import ParamScalersField
from stream_ml.core.setup_package import BACKGROUND_KEY
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.cached_property import cached_property
from stream_ml.core.utils.frozen_dict import FrozenDict
from stream_ml.core.utils.funcs import get_prefixed_kwargs
from stream_ml.core.utils.scale import DataScaler  # noqa: TCH001
from stream_ml.core.utils.sentinel import MISSING

__all__: list[str] = []

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stream_ml.core.data import Data


@dataclass
class MixtureModel(ModelsBase[Array, NNModel]):
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
    data_scaler: DataScaler

    # Coordinates, indpendent and dependent.
    indep_coord_names: tuple[str, ...] = ("phi1",)

    # Model Parameters, generally produced by the neural network.
    # param_names is a cached property.
    param_bounds: MixtureParamBoundsField[Array] = MixtureParamBoundsField[Array](
        ParamBounds()
    )
    param_scalers: ParamScalersField[Array] = ParamScalersField()
    # TODO! Have Identity as the default

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

        # Add scaling to the param bounds  # TODO! unfreeze then freeze
        for k, v in self.param_bounds.items():
            if not isinstance(k, str):
                raise TypeError

            if not isinstance(v, FrozenDict):
                self.param_bounds._dict[k] = replace(v, scaler=self.param_scalers[k])
                continue
            for k2, v2 in v.items():
                v._dict[k2] = replace(v2, scaler=self.param_scalers[k, k2])

    @cached_property
    def param_names(self) -> ParamNames:  # type: ignore[override]
        """Parameter names."""
        names: list[str | tuple[str, tuple[str, ...]]] = []
        for c, m in self.components.items():
            names.append(f"{c}.weight")
            names.extend(
                (f"{c}.{p[0]}", p[1]) if isinstance(p, tuple) else f"{c}.{p}"
                for p in m.param_names
            )
        return ParamNames(names)

    @cached_property
    def mixture_param_names(self) -> ParamNames:
        """Mixture parameter names."""
        return ParamNames([f"{c}.weight" for c in self.components])

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
        j: int = 0
        for n, m in self.components.items():  # iter thru models
            # Weight
            if n != BACKGROUND_KEY:
                weight = arr[:, j : j + 1]
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
                    start=self.xp.zeros((len(arr), 1), dtype=arr.dtype),
                )

            # Add the component's parameters, prefixed with the component name
            pars[n + ".weight"] = weight
            j += 1  # Increment the index (weight)

            # Parameters
            if len(m.param_names.flat) == 0:
                continue
            marr = arr[:, slice(j, j + len(m.param_names.flat))]
            pars.update(m.unpack_params_from_arr(marr).add_prefix(n + "."))
            j += len(m.param_names.flat)  # Increment the index (parameters)

        # Allow for conversation between components
        for hook in self.unpack_params_hooks:
            pars = hook(pars)

        return Params(pars)

    # ===============================================================

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Log likelihood.

        Just the log-sum-exp of the individual log-likelihoods.

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
        # Get the parameters for each model, stripping the model name,
        # and use that to evaluate the log likelihood for the model.
        lnliks = tuple(
            self.xp.log(mpars[(f"{name}.weight",)])
            + model.ln_likelihood(
                mpars.get_prefixed(name),
                data,
                **get_prefixed_kwargs(name, kwargs),
            )
            for name, model in self.components.items()
        )
        # Sum over the models, keeping the data dimension
        return self.xp.logsumexp(self.xp.hstack(lnliks), 1)[:, None]
