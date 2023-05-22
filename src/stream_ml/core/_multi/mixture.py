"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Literal, cast, overload

from stream_ml.core import NNField
from stream_ml.core._multi.bases import ModelsBase
from stream_ml.core.params import Params, add_prefix, freeze_params
from stream_ml.core.params._field import ModelParametersField
from stream_ml.core.setup_package import BACKGROUND_KEY
from stream_ml.core.typing import Array, NNModel
from stream_ml.core.utils.funcs import get_prefixed_kwargs
from stream_ml.core.utils.scale import DataScaler  # noqa: TCH001
from stream_ml.core.utils.sentinel import MISSING

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.typing import ParamNameAllOpts, ParamsLikeDict


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
                    m.unpack_params_from_arr(
                        marr,
                        extras=extras_ | {"weight": weight},  # pass the weight
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
