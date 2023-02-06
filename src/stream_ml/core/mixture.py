"""Core feature."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Callable, cast

from stream_ml.core.bases import ModelsBase
from stream_ml.core.data import Data
from stream_ml.core.params import ParamNames, Params
from stream_ml.core.setup_package import BACKGROUND_KEY, WEIGHT_NAME
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDictField

__all__: list[str] = []


@dataclass
class MixtureModel(ModelsBase[Array]):
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

    tied_params : Mapping[str, Callable[[Params], Array]], optional keyword-only
        Mapping of parameter names to functions that take the parameters of the
        model and return the value of the tied parameter. This is useful for
        tying parameters across models, e.g. the background and stream models
        in a mixture model.

    priors : tuple[PriorBase, ...], optional keyword-only
        Mapping of parameter names to priors. This is useful for setting priors
        on parameters across models, e.g. the background and stream models in a
        mixture model.
    """

    _: KW_ONLY
    tied_params: FrozenDictField[
        str, Callable[[Mapping[str, Array | Mapping[str, Array]]], Array]
    ] = FrozenDictField({})

    def __post_init__(self) -> None:
        # Add the param_names  # TODO: make sure no duplicates
        self._param_names: ParamNames = ParamNames(
            (f"{c}.{p[0]}", p[1]) if isinstance(p, tuple) else f"{c}.{p}"
            for c, m in self.components.items()
            for p in m.param_names
        )

        # Check if the model has a background component.
        # If it does, then it must be the last component.
        includes_bkg = BACKGROUND_KEY in self.components
        if includes_bkg and tuple(self.components.keys())[-1] != BACKGROUND_KEY:
            msg = "the background model must be the last component."
            raise KeyError(msg)
        self._includes_bkg: bool = includes_bkg

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
        j = 0
        for n, m in self.components.items():  # iter thru models
            # Get relevant parameters by index
            mp_arr = p_arr[:, slice(j, j + len(m.param_names.flat))]

            if n == BACKGROUND_KEY:
                # The background is special, because it has a weight parameter
                # that is defined as 1 - the sum of the other weights.
                # So, we need to calculate the background weight from the
                # other weights. Note that the background weight can be included
                # in the parameter array, but it should not be determined by
                # any network output, rather just a placeholder.

                # The background weight is 1 - the other weights
                bkg_weight: Array = 1 - sum(
                    (
                        cast("Array", pars[f"{k}.weight"])
                        for k in tuple(self.components.keys())[:-1]
                        # skipping the background, which is the last component
                    ),
                    start=self.xp.zeros((len(mp_arr), 1), dtype=mp_arr.dtype),
                )
                # It is the index-0 column of the array
                mp_arr = self.xp.hstack((bkg_weight, mp_arr[:, 1:]))

            # Skip empty (and incrementing the index)
            if mp_arr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars.update(m.unpack_params_from_arr(mp_arr).add_prefix(n + "."))

            # Increment the index
            j += len(m.param_names.flat)

        # Always add the combined weight
        pars[WEIGHT_NAME] = cast(
            "Array", sum(cast("Array", pars[f"{k}.weight"]) for k in self.components)
        )

        # Add / update the dependent parameters
        for name, tie in self.tied_params.items():
            pars[name] = tie(pars)

        return Params[Array](pars)

    # ===============================================================

    def ln_likelihood_arr(
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
            model.ln_likelihood_arr(
                mpars.get_prefixed(name),
                data,
                **self._get_prefixed_kwargs(name, kwargs),
            )
            for name, model in self.components.items()
        )
        # Sum over the models, keeping the data dimension
        return self.xp.logsumexp(self.xp.hstack(lnliks), 1)[:, None]
