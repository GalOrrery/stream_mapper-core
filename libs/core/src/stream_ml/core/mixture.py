"""Core feature."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Callable, Literal, cast

# LOCAL
from stream_ml.core.api import WEIGHT_NAME
from stream_ml.core.bases import ModelsBase
from stream_ml.core.params import ParamNames, Params
from stream_ml.core.typing import Array
from stream_ml.core.utils.frozen_dict import FrozenDictField

if TYPE_CHECKING:
    # LOCAL
    pass

__all__: list[str] = []


BACKGROUND_KEY = "background"


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

    @abstractmethod
    def _hook_unpack_bkg_weight(
        self, weight: Array | Literal[1], mp_arr: Array
    ) -> Array:
        """Hook to unpack the background weight.

        This is necessary because JAX doesn't support assignment to a slice.
        """
        raise NotImplementedError

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
            if n == BACKGROUND_KEY:  # include the weight
                weight = sum(
                    cast("Array", pars[f"{k}.weight"])
                    for k in tuple(self.components.keys())[:-1]
                    # skipping the background, which is the last component
                )
                bkg_weight: Array | Literal[1] = (
                    1 - weight if not isinstance(weight, int) else 1
                )

                # The background weight is 1 - the other weights
                # it is the index-0 column of the array
                mp_arr = self._hook_unpack_bkg_weight(bkg_weight, mp_arr)

            # Skip empty (and incrementing the index)
            if mp_arr.shape[1] == 0:
                continue

            # Add the component's parameters, prefixed with the component name
            pars.update(m.unpack_params_from_arr(mp_arr).add_prefix(n + "."))

            # Increment the index
            j += len(m.param_names.flat)

        # Always add the combined weight
        pars[WEIGHT_NAME] = cast(
            "Array", sum(pars[f"{k}.weight"] for k in self.components)
        )

        # Add / update the dependent parameters
        for name, tie in self.tied_params.items():
            pars[name] = tie(pars)

        return Params[Array](pars)
