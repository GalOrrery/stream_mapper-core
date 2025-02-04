"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, InitVar, dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from stream_mapper.core._api import SupportsXP
from stream_mapper.core._connect.xp_namespace import XP_NAMESPACE
from stream_mapper.core.params.scaler import ParamScaler  # noqa: TC001
from stream_mapper.core.typing import Array, ArrayNamespace, ParamNameTupleOpts
from stream_mapper.core.utils import array_at, within_bounds
from stream_mapper.core.utils.dataclasses import ArrayNamespaceReprMixin

if TYPE_CHECKING:
    from collections.abc import Iterator

    from stream_mapper.core import Data, ModelAPI as Model, Params
    from stream_mapper.core.typing import NNModel

    Self = TypeVar("Self", bound="ParameterBounds")  # type: ignore[type-arg]


@dataclass(frozen=True, repr=False)
class ParameterBounds(
    ArrayNamespaceReprMixin[Array], SupportsXP[Array], metaclass=ABCMeta
):
    """Base class for prior bounds."""

    lower: Array | float
    upper: Array | float

    eps: Array | float | None = None

    _: KW_ONLY
    param_name: ParamNameTupleOpts | None = None
    scaler: InitVar[ParamScaler[Array] | None] = None
    name: str | None = None  # the name of the prior
    neg_inf: Array | float = -float("inf")

    array_namespace: ArrayNamespace[Array]

    def __new__(
        cls: type[Self],
        *args: Any,  # noqa: ARG003
        array_namespace: ArrayNamespace[Array] | str | None = None,
        **kwargs: Any,  # noqa: ARG003
    ) -> Self:
        # Create the instance
        self = super().__new__(cls)

        # Set the array namespace
        xp: ArrayNamespace[Array] | None = XP_NAMESPACE[
            (
                getattr(cls, "array_namespace", None)
                if array_namespace is None
                else array_namespace
            )
        ]
        object.__setattr__(self, "array_namespace", xp)

        return self

    def __post_init__(self, scaler: ParamScaler[Array] | None) -> None:
        """Post-init."""
        if self.lower >= self.upper:
            msg = "lower must be less than upper"
            raise ValueError(msg)

        # Need to convert xp if it's a string
        if isinstance(self.array_namespace, str):
            object.__setattr__(
                self, "array_namespace", XP_NAMESPACE[self.array_namespace]
            )

        # Scale the bounds. Note that we add and subtract eps to the bounds to
        # ensure that the bounds are not violated when the parameters are
        # scaled. Why 2.5? Because the eps gets doubled when the parameters are
        # scaled in Sigmoid, so we need to account for that.
        self._scaled_bounds: tuple[Array, Array]
        if scaler is not None:
            lower = scaler.transform(self.lower)
            eps = 2.5 * self.xp.finfo(getattr(lower, "dtype", float)).eps
            object.__setattr__(
                self,
                "_scaled_bounds",
                (lower + eps, scaler.transform(self.upper) - eps),
            )

    # =========================================================================

    def logpdf(
        self,
        mpars: Params[Array],
        data: Data[Array],
        model: Model[Array, NNModel],
        current_lnpdf: Array | None = None,
        /,
    ) -> Array:
        """Evaluate the logpdf.

        This log-pdf is added to the current logpdf. So if you want to set the
        logpdf to a specific value, you can uses the `current_lnpdf` to set the
        output value such that ``current_lnpdf + logpdf = <want>``.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array], position-only
            The data for which evaluate the prior.
        model : Model, position-only
            The model for which evaluate the prior.
        current_lnpdf : Array | None, optional position-only
            The current logpdf, by default `None`. This is useful for setting
            the additive log-pdf to a specific value.

        Returns
        -------
        Array
        """
        if self.param_name is None:
            msg = "need to set param_name"
            raise ValueError(msg)

        bp = self.xp.zeros_like(mpars[self.param_name])
        return array_at(
            bp, ~within_bounds(mpars[self.param_name], self.lower, self.upper)
        ).set(self.neg_inf)

    @abstractmethod
    def __call__(
        self, pred: Array, data: Data[Array], model: Model[Array, NNModel], /
    ) -> Array:
        """Evaluate the forward step in the prior.

        Parameters
        ----------
        pred : Array, position-only
            The input to evaluate the prior at.
        data : Array, position-only
            The data to evaluate the prior at.
        model : `~stream_mapper.core.Model`, position-only
            The model to evaluate the prior at.

        Returns
        -------
        Array
        """
        ...

    # =========================================================================

    @property
    def bounds(self) -> tuple[Array | float, Array | float]:
        """Get the bounds."""
        return (self.lower, self.upper)

    @property
    def scaled_bounds(self) -> tuple[Array | float, Array | float]:
        """Get the scaled bounds."""
        if not hasattr(self, "_scaled_bounds"):
            msg = "need to pass scaler to prior bounds"
            raise ValueError(msg)
        return self._scaled_bounds

    # =========================================================================

    def __iter__(self) -> Iterator[Array | float]:
        """Iterate over the bounds."""
        yield from self.bounds
