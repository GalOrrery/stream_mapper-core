"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from abc import ABCMeta
from dataclasses import KW_ONLY, dataclass, fields
from functools import reduce
from textwrap import indent
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from stream_ml.core._connect.nn_namespace import NN_NAMESPACE
from stream_ml.core._connect.xp_namespace import XP_NAMESPACE
from stream_ml.core._core.field import NNField
from stream_ml.core._core.model_api import Model
from stream_ml.core.params import ModelParametersField, Params, freeze_params, set_param
from stream_ml.core.setup_package import CompiledShim
from stream_ml.core.typing import Array, ArrayNamespace, BoundsT, NNModel, NNNamespace
from stream_ml.core.utils import DataScaler, within_bounds
from stream_ml.core.utils.dataclasses import ArrayNamespaceReprMixin
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField

if TYPE_CHECKING:
    from stream_ml.core import Data
    from stream_ml.core.prior import Prior
    from stream_ml.core.typing import ParamNameAllOpts, ParamsLikeDict

    Self = TypeVar("Self", bound="ModelBase[Array, NNModel]")  # type: ignore[valid-type]


#####################################################################


@dataclass(unsafe_hash=True, repr=False)
class ModelBase(
    Model[Array, NNModel],
    ArrayNamespaceReprMixin[Array],
    CompiledShim,
    metaclass=ABCMeta,
):
    """Single-model base class.

    Parameters
    ----------
    net : NNField[NNModel], keyword-only
        The neural network.

    indep_coord_names : tuple[str, ...], optional keyword-only
        The names of the independent coordinates, e.g. "phi1". Default is
        ("phi1",).
    coord_names : tuple[str, ...], keyword-only
        The names of the coordinates, not including the 'independent' variable.
        E.g. for independent variable 'phi1' this might be ('phi2', 'prlx',
        ...).
    coord_err_names : tuple[str, ...] | None, optional
        Coordinate error names, e.g. "phi2_err", "pm_phi1_err". Default is
        `None`, which means that no coordinate errors are used. If specified,
        must have one entry per coordinate in `coord_names`, in the same order.
    coord_bounds : Mapping[str, tuple[float, float]], keyword-only
        The bounds on the coordinates. If not provided, the bounds are (-inf,
        inf) for all coordinates.

    params : ModelParameters[Array], optional keyword-only
        Model parameters. Default is empty.

    priors: tuple[Prior[Array], ...], optional keyword-only
        Priors on the parameters. Default is empty.

    data_scaler : DataScaler[Array], keyword-only
        The data scaler.

    require_where: bool, optional keyword-only
        Whether the model requires the `where` keyword. Default is `False`.

    array_namespace : ArrayNamespace[Array], keyword-only
        The array namespace.
    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).
    """

    net: NNField[NNModel, NNModel] = NNField()

    _: KW_ONLY
    array_namespace: ArrayNamespace[Array]
    name: str | None = None  # the name of the model

    # Data scaling
    data_scaler: DataScaler[Array]

    # Coordinates, indpendent and dependent.
    indep_coord_names: tuple[str, ...] = ("phi1",)
    coord_names: tuple[str, ...]
    coord_err_names: tuple[str, ...] | None = None
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())

    # Model Parameters, generally produced by the neural network.
    params: ModelParametersField[Array] = ModelParametersField[Array]()

    # Priors on the parameters.
    priors: tuple[Prior[Array], ...] = ()

    # Masked data
    require_where: bool = True

    def __new__(
        cls: type[Self],
        *args: Any,  # noqa: ARG003
        array_namespace: ArrayNamespace[Array] | str | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> Self:
        # Construct the dataclass. Need to use `__new__` to ensure that the
        # array (xp) and nn (xpnn) namespaces are available to the dataclass
        # in the stanard initialization.

        # Create the model instance.
        # TODO: Model.__new__ over objects.__new__ is a mypyc hack.
        self = Model.__new__(cls)

        # Ensure that the array and nn namespaces are available to the dataclass
        # descriptor fields.
        xp: ArrayNamespace[Array] | None = XP_NAMESPACE[
            (
                getattr(cls, "array_namespace", None)
                if array_namespace is None
                else array_namespace
            )
        ]
        if xp is None:
            msg = f"Model {cls} requires array_namespace"
            raise TypeError(msg)
        object.__setattr__(self, "array_namespace", xp)
        object.__setattr__(self, "_nn_namespace_", NN_NAMESPACE[xp])

        return self

    def __post_init__(self) -> None:
        """Post-init validation."""
        super().__post_init__()
        self._mypyc_init_descriptor()  # TODO: Remove this when mypyc is fixed.

        # Have to reset array_namespace if it's a string.
        if isinstance(self.array_namespace, str):
            object.__setattr__(
                self,
                "array_namespace",
                XP_NAMESPACE[self.array_namespace],
            )

        # Type hint
        self._nn_namespace_: NNNamespace[NNModel, Array]

        # Coordinate bounds are necessary (before they were auto-filled).
        if self.coord_bounds.keys() != set(self.coord_names):
            msg = (
                f"`coord_bounds` ({tuple(self.coord_bounds.keys())}) do not match "
                f"`coord_names` ({self.coord_names})."
            )
            raise ValueError(msg)

        # coord_err_names must be None or the same length as coord_names.
        # we can't check that the names are the same, because they aren't.
        # TODO: better way to ensure that
        kcen = self.coord_err_names
        if kcen is not None and len(kcen) != self.ndim:
            msg = (
                f"`coord_err_names` ({kcen}) must be None or "
                f"the same length as `coord_names` ({self.coord_names})."
            )
            raise ValueError(msg)

        # Parameters must be a subset of the `coord_names`.
        if not set(self.params.keys()).issubset(self.coord_names):
            msg = (
                f"`params` ({tuple(self.params.keys())}) must be a subset of "
                f"`coord_names` ({self.coord_names})."
            )
            raise ValueError(msg)

    # ========================================================================

    def _stack_param(self, p: Params[Array], k: str, cns: tuple[str, ...], /) -> Array:
        return self.xp.stack(tuple(p[(c, k)] for c in cns), 1)

    # ========================================================================

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

        This function takes the NN output array and unpacks it into a dictionary
        with the parameter names as keys.

        Parameters
        ----------
        arr : Array, positional-only
            Parameter array.
        extras : dict[ParamNameAllOpts, Array] | None, keyword-only
            Extra arrays to add.
        freeze : bool, optional keyword-only
            Whether to freeze the parameters. Default is `True`.

        Returns
        -------
        Params[Array]
        """
        pars: ParamsLikeDict[Array] = {}
        k: ParamNameAllOpts
        for i, (k, p) in enumerate(self.params.flatsitems()):
            # Unscale and set in the nested dict structure
            set_param(pars, k, p.scaler.inverse_transform(arr[:, i]))

        for k, v in (extras or {}).items():
            set_param(pars, k, v)

        return freeze_params(pars) if freeze else pars

    # ========================================================================
    # Statistics

    def _ln_prior_coord_bnds(self, data: Data[Array], /) -> Array:
        """Elementwise log prior for coordinate bounds.

        Zero everywhere except where the data are outside the
        coordinate bounds, where it is -inf.
        """
        shape = data.array.shape[:1] + data.array.shape[2:]
        where = reduce(
            self.xp.logical_or,
            (
                ~within_bounds(data[k], *v)
                for k, v in self.coord_bounds.items()
                if k in data.names
                # don't require all coordinates to be present in the data,
                # e.g. "distmod" on an isochrone model.
            ),
            self.xp.zeros(shape, dtype=bool),
        )
        return self.xp.where(
            where,
            self.xp.full(shape, -self.xp.inf),
            self.xp.zeros(shape),
        )

    def ln_prior(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array[(N,F)]]
            Data (phi1, phi2).

        Returns
        -------
        Array
        """
        # Coordinate Bounds
        lnp = self._ln_prior_coord_bnds(data)
        # Parameter Bounds
        for p in self.params.flatvalues():
            lnp = lnp + p.bounds.logpdf(mpars, data, self, lnp)
        # Priors
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp)

        return lnp

    def ln_evidence(self, data: Data[Array], /) -> Array:
        """Log evidence.

        Parameters
        ----------
        data : Data[Array[(N,F)]], positional-only
            Data (phi1, phi2).

        Returns
        -------
        Array
        """
        return self.xp.zeros(len(data))

    # ========================================================================
    # ML

    def _forward_priors(self, out: Array, scaled_data: Data[Array], /) -> Array:
        """Forward pass.

        Parameters
        ----------
        out : Array, positional-only
            Input.
        scaled_data : Data[Array[(N,F)]], positional-only
            Data, scaled by ``data_scaler``.

        Returns
        -------
        Array
            Same as input.
        """
        # Parameter bounds
        for p in self.params.flatvalues():
            out = p.bounds(out, scaled_data, self)

        # Other priors  # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            out = prior(out, scaled_data, self)
        return out

    # ========================================================================
    # Misc

    def __str__(self) -> str:
        """Return nicer string representation."""
        fs = (
            indent(f"{f.name}: {getattr(self, f.name)!s}", prefix="\t")
            for f in fields(self)
        )
        return self.__class__.__name__ + "(\n" + "\n".join(fs) + "\n)"
