"""Core feature."""

from __future__ import annotations

import textwrap
from abc import ABCMeta
from dataclasses import KW_ONLY, InitVar, dataclass, fields
from functools import reduce
from math import inf
from typing import TYPE_CHECKING, ClassVar

from stream_ml.core.api import WEIGHT_NAME, Model
from stream_ml.core.data import Data
from stream_ml.core.params import ParamBounds, Params, freeze_params, set_param
from stream_ml.core.params.bounds import ParamBoundsField
from stream_ml.core.params.names import ParamNamesField
from stream_ml.core.prior.base import PriorBase
from stream_ml.core.setup_package import CompiledShim
from stream_ml.core.typing import Array, ArrayNamespace, BoundsT
from stream_ml.core.utils.compat import array_at
from stream_ml.core.utils.frozen_dict import FrozenDict, FrozenDictField
from stream_ml.core.utils.funcs import within_bounds

if TYPE_CHECKING:
    pass

__all__: list[str] = []


@dataclass(unsafe_hash=True)
class ModelBase(Model[Array], CompiledShim, metaclass=ABCMeta):
    """Single-model base class.

    Parameters
    ----------
    name : str or None, optional keyword-only
        The (internal) name of the model, e.g. 'stream' or 'background'. Note
        that this can be different from the name of the model when it is used in
        a mixture model (see :class:`~stream_ml.core.core.MixtureModel`).

    coord_names : tuple[str, ...], keyword-only
        The names of the coordinates, not including the 'independent' variable.
        E.g. for independent variable 'phi1' this might be ('phi2', 'prlx',
        ...).
    param_names : `~stream_ml.core.params.ParamNames`, keyword-only
        The names of the parameters. Parameters dependent on the coordinates are
        grouped by the coordinate name.
        E.g. ('weight', ('phi1', ('mu', 'sigma'))).

    coord_bounds : Mapping[str, tuple[float, float]], keyword-only
        The bounds on the coordinates. If not provided, the bounds are
        (-inf, inf) for all coordinates.

    param_bounds : `~stream_ml.core.params.ParamBounds`, keyword-only
        The bounds on the parameters.
    """

    indep_coord_name: str = "phi1"  # TODO: move up class hierarchy?

    _: KW_ONLY
    array_namespace: InitVar[ArrayNamespace[Array]]
    name: str | None = None  # the name of the model

    coord_names: tuple[str, ...]
    param_names: ParamNamesField = ParamNamesField()

    # Bounds on the coordinates and parameters.
    coord_bounds: FrozenDictField[str, BoundsT] = FrozenDictField(FrozenDict())
    param_bounds: ParamBoundsField[Array] = ParamBoundsField[Array](ParamBounds())

    priors: tuple[PriorBase[Array], ...] = ()

    DEFAULT_BOUNDS: ClassVar  # TODO: [PriorBounds[Any]]

    def __post_init__(self, array_namespace: ArrayNamespace[Array]) -> None:
        """Post-init validation."""
        super().__post_init__(array_namespace=array_namespace)
        self._init_descriptor()  # TODO: Remove this when mypyc is fixed.

        self._array_namespace_ = array_namespace

        # Validate the param_names
        if not self.param_names:
            msg = "param_names must be specified"
            raise ValueError(msg)

        # Make coord bounds if not provided
        crnt_cbs = dict(self.coord_bounds)
        cbs = {n: crnt_cbs.pop(n, (-inf, inf)) for n in self.coord_names}
        if crnt_cbs:  # Error if there are extra keys
            msg = f"coord_bounds contains invalid keys {crnt_cbs.keys()}."
            raise ValueError(msg)
        self.coord_bounds = FrozenDict(cbs)

        # Make parameter bounds
        # 1) Make the default bounds for all parameters.
        # 2) Update from the user-specified bounds.
        # 3) Fix up the names so each bound references its parameter.
        self.param_bounds: ParamBounds[Array] = (
            ParamBounds.from_names(self.param_names, default=self.DEFAULT_BOUNDS)
            | self.param_bounds
        )
        self.param_bounds._fixup_param_names()
        # Validate param bounds.
        self.param_bounds.validate(self.param_names)

    @property
    def xp(self) -> ArrayNamespace[Array]:
        """Array namespace."""
        return self._array_namespace_

    # ========================================================================

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
        pars: dict[str, Array | dict[str, Array]] = {}
        for i, k in enumerate(self.param_names.flats):
            set_param(pars, k, p_arr[:, i : i + 1])
        return freeze_params(pars)

    # ========================================================================
    # Statistics

    def _ln_prior_coord_bnds(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Elementwise log prior for coordinate bounds.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
            Zero everywhere except where the data are outside the
            coordinate bounds, where it is -inf.
        """
        lnp = self.xp.zeros_like(mpars[(WEIGHT_NAME,)])
        where = reduce(
            self.xp.logical_or,
            (~within_bounds(data[k], *v) for k, v in self.coord_bounds.items()),
        )
        return array_at(lnp, where).set(-self.xp.inf)

    def ln_prior_arr(
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
        lnp = lnp + self._ln_prior_coord_bnds(mpars, data)
        # Parameter Bounds
        for bounds in self.param_bounds.flatvalues():
            lnp = lnp + bounds.logpdf(mpars, data, self, lnp)
        # Priors
        for prior in self.priors:
            lnp = lnp + prior.logpdf(mpars, data, self, lnp)

        return lnp

    # ========================================================================
    # ML

    def _forward_priors(self, out: Array, data: Data[Array]) -> Array:
        """Forward pass.

        Parameters
        ----------
        out : Array
            Input.
        data : Data[Array]
            Data.

        Returns
        -------
        Array
            Same as input.
        """
        # Parameter bounds  # FIXME!
        for bnd in self.param_bounds.flatvalues():
            out = bnd(out, data, self)

        # Other priors
        # TODO: a better way to do the order of the priors.
        for prior in self.priors:
            out = prior(out, data, self)
        return out

    # ========================================================================
    # Misc

    def __str__(self) -> str:
        """String representation."""
        s = f"{self.__class__.__name__}(\n"
        s += "\n".join(
            textwrap.indent(f"{f.name}: {getattr(self, f.name)!s}", prefix="\t")
            for f in fields(self)
        )
        s += "\n)"
        return s
