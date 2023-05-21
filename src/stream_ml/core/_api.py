"""Core feature."""

from __future__ import annotations

__all__: list[str] = []

from typing import TYPE_CHECKING, Any, Protocol

from stream_ml.core.typing import Array, ArrayNamespace, NNModel

if TYPE_CHECKING:
    from stream_ml.core.data import Data
    from stream_ml.core.params._values import Params
    from stream_ml.core.typing import NNNamespace


class SupportsXP(Protocol[Array]):
    """Protocol for objects that support array namespaces."""

    array_namespace: ArrayNamespace[Array]

    @property
    def xp(self) -> ArrayNamespace[Array]:
        """Array namespace."""
        return self.array_namespace


class SupportsXPNN(SupportsXP[Array], Protocol[Array, NNModel]):
    """Protocol for objects that support array and NN namespaces."""

    _nn_namespace_: NNNamespace[NNModel, Array]

    @property
    def xpnn(self) -> NNNamespace[NNModel, Array]:
        """NN namespace."""
        return self._nn_namespace_


class HasName(Protocol):
    """Protocol for objects that have a name."""

    name: str | None


#####################################################################
# Probabilities


class LnProbabilities(Protocol[Array]):
    """Protocol for objects that support probabilities."""

    def ln_likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Any
    ) -> Array:
        """Elementwise log-likelihood of the model.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Array
        """
        ...

    def ln_prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Elementwise log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        current_lnp : Array | None, optional
            Current value of the log prior, by default `None`.

        Returns
        -------
        Array
        """
        ...

    def ln_posterior(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Elementwise log posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.
        **kw : Array
            Arguments.

        Returns
        -------
        Array
        """
        return self.ln_likelihood(mpars, data, **kw) + self.ln_prior(mpars, data)


class TotalLnProbabilities(Protocol[Array]):
    """Protocol for objects that support total probabilities."""

    def ln_likelihood_tot(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Total log-likelihood of the model.

        This is evaluated over the entire data set.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        ...

    def ln_prior_tot(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Log prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.

        Returns
        -------
        Array
        """
        ...

    def ln_posterior_tot(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Log posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kw : Array
            Keyword arguments. These are passed to the likelihood function.

        Returns
        -------
        Array
        """
        ...


class Probabilities(Protocol[Array]):
    """Protocol for objects that support probabilities."""

    def likelihood(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Elementwise likelihood of the model.

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
        ...

    def prior(
        self, mpars: Params[Array], data: Data[Array], current_lnp: Array | None = None
    ) -> Array:
        """Elementwise prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        current_lnp : Array | None, optional
            Current value of the log prior, by default `None`.

        Returns
        -------
        Array
        """
        ...

    def posterior(self, mpars: Params[Array], data: Data[Array], **kw: Array) -> Array:
        """Elementwise posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kw : Array
            Arguments.

        Returns
        -------
        Array
        """
        ...


class TotalProbabilities(Protocol[Array]):
    """Protocol for objects that support total probabilities."""

    def likelihood_tot(
        self, mpars: Params[Array], data: Data[Array], **kwargs: Array
    ) -> Array:
        """Total likelihood of the model.

        This is evaluated over the entire data set.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.
        **kwargs : Array
            Additional arguments.

        Returns
        -------
        Array
        """
        ...

    def prior_tot(self, mpars: Params[Array], data: Data[Array]) -> Array:
        """Total prior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data
            Data.

        Returns
        -------
        Array
        """
        ...

    def posterior_tot(
        self, mpars: Params[Array], data: Data[Array], **kw: Array
    ) -> Array:
        """Total posterior.

        Parameters
        ----------
        mpars : Params[Array], positional-only
            Model parameters. Note that these are different from the ML
            parameters.
        data : Data[Array]
            Data.
        **kw : Array
            Keyword arguments. These are passed to the likelihood function.

        Returns
        -------
        Array
        """
        ...
