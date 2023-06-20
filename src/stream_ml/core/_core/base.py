"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from abc import ABCMeta
from dataclasses import KW_ONLY, dataclass, fields
from functools import reduce
from math import inf
import textwrap
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, cast, overload

from stream_ml.core._core.api import Model
from stream_ml.core._core.field import NNField
from stream_ml.core.params._field import ModelParametersField
from stream_ml.core.params._values import Params, freeze_params, set_param
from stream_ml.core.setup_package import CompiledShim
from stream_ml.core.typing import Array, ArrayNamespace, BoundsT, NNModel
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField
from stream_ml.core.utils.funcs import within_bounds
from stream_ml.core.utils.scale._api import DataScaler  # noqa: TCH001

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.prior import PriorBase
    from stream_ml.core.typing import NNNamespace, ParamNameAllOpts, ParamsLikeDict

    Self = TypeVar("Self", bound="ModelBase[Array, NNModel]")  # type: ignore[valid-type]  # noqa: E501


#####################################################################


class NNNamespaceMap(Protocol):
    """Protocol for mapping array namespaces to NN namespaces."""

    def __getitem__(self, key: ArrayNamespace[Array]) -> NNNamespace[NNModel, Array]:
        """Get item."""
        ...

    def __setitem__(
        self, key: ArrayNamespace[Array], value: NNNamespace[NNModel, Array]
    ) -> None:
        """Set item."""
        ...


NN_NAMESPACE = cast(NNNamespaceMap, {})


#####################################################################


@dataclass(unsafe_hash=True)
class ModelBase(Model[Array, NNModel], CompiledShim, metaclass=ABCMeta):
    """Single-model base class.

    Parameters
    ----------
    net : NNField[NNModel], keyword-only
        The neural network.

    array_namespace : ArrayNamespace[Array], keyword-only
        The array namespace.

    coord_names : tuple[str, ...], keyword-only
        The names of the coordinates, not including the 'independent' variable.
        E.g. for independent variable 'phi1' this might be ('phi2', 'prlx',
        ...).
    coord_bounds : Mapping[str, tuple[float, float]], keyword-only
        The bounds on the coordinates. If not provided, the bounds are
        (-inf, inf) for all coordinates.

    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).
    """

    net: NNField[NNModel] = NNField()

    _: KW_ONLY
    array_namespace: ArrayNamespace[Array]
    name: str | None = None  # the name of the model

    # Standardizer
    data_scaler: DataScaler

    # Coordinates, indpendent and dependent.
    indep_coord_names: tuple[str, ...] = ("phi1",)
    coord_names: tuple[str, ...]
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())
    coord_err_names: tuple[str, ...] | None = None

    # Model Parameters, generally produced by the neural network.
    params: ModelParametersField[Array] = ModelParametersField[Array]()

    # Priors on the parameters.
    priors: tuple[PriorBase[Array], ...] = ()

    def __new__(
        cls: type[Self],
        *args: Any,  # noqa: ARG003
        array_namespace: ArrayNamespace[Array] | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> Self:
        # Create the model instance.
        # TODO: Model.__new__ over objects.__new__ is a mypyc hack.
        self = Model.__new__(cls)

        # Ensure that the array and nn namespaces are available to the dataclass
        # descriptor fields.
        xp: ArrayNamespace[Array] | None = (
            getattr(cls, "array_namespace", None)
            if array_namespace is None
            else array_namespace
        )
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

        # Make coord bounds if not provided
        crnt_cbs = dict(self.coord_bounds)
        cbs = {n: crnt_cbs.pop(n, (-inf, inf)) for n in self.coord_names}
        if crnt_cbs:  # Error if there are extra keys
            msg = f"coord_bounds contains invalid keys {crnt_cbs.keys()}."
            raise ValueError(msg)
        object.__setattr__(self, "coord_bounds", FrozenDict(cbs))

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
        lnp = self.xp.zeros(data.array.shape[:1] + data.array.shape[2:])
        where = reduce(
            self.xp.logical_or,
            (~within_bounds(data[k], *v) for k, v in self.coord_bounds.items()),
            self.xp.zeros(lnp.shape, dtype=bool),
        )
        return array_at(lnp, where).set(-self.xp.inf)

    def ln_prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data (phi1, phi2).
        current_lnp : Array | None, optional
            Current value of the log prior, by default `None`.

        Returns
        -------
        Array
        """
        lnp: Array = self.xp.zeros(()) if current_lnp is None else current_lnp

        # Coordinate Bounds
        lnp = lnp + self._ln_prior_coord_bnds(data)
        # Parameter Bounds
        for p in self.params.flatvalues():
            lnp = lnp + p.bounds.logpdf(mpars, data, self, lnp, xp=self.xp)
        # Priors
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp, xp=self.xp)

        return lnp

    # ========================================================================
    # ML

    def _forward_priors(self, out: Array, scaled_data: Data[Array], /) -> Array:
        """Forward pass.

        Parameters
        ----------
        out : Array, positional-only
            Input.
        scaled_data : Data[Array], positional-only
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
        s = f"{self.__class__.__name__}(\n"
        s += "\n".join(
            textwrap.indent(f"{f.name}: {getattr(self, f.name)!s}", prefix="\t")
            for f in fields(self)
        )
        s += "\n)"
        return s
