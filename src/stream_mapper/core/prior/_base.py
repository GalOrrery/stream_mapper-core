"""Core feature."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from abc import ABCMeta, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from stream_mapper.core._api import SupportsXP
from stream_mapper.core._connect.xp_namespace import XP_NAMESPACE
from stream_mapper.core.typing import Array
from stream_mapper.core.utils.dataclasses import ArrayNamespaceReprMixin

if TYPE_CHECKING:
    from stream_mapper.core import Data, ModelAPI as Model, Params
    from stream_mapper.core.typing import ArrayNamespace, NNModel


Self = TypeVar("Self", bound="Prior[Array]")  # type: ignore[valid-type]


@dataclass(frozen=True, repr=False)
class Prior(ArrayNamespaceReprMixin[Array], SupportsXP[Array], metaclass=ABCMeta):
    """Prior."""

    _: KW_ONLY
    name: str | None = None  # the name of the prior
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
        if xp is None:
            msg = f"Model {cls} requires array_namespace"
            raise TypeError(msg)
        object.__setattr__(self, "array_namespace", xp)

        return self

    def __post_init__(self, *args: Any, **kwargs: Any) -> None:
        """Post-init."""
        if isinstance(self.array_namespace, str):
            object.__setattr__(self, "array_namespace", XP_NAMESPACE[self.xp])

    @abstractmethod
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
            The logpdf.
        """
        ...

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
        return pred
