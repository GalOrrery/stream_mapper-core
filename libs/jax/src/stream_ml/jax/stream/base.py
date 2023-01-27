"""Base Stream Model class."""

from __future__ import annotations

# STDLIB
from dataclasses import dataclass

# LOCAL
from stream_ml.core.data import Data
from stream_ml.jax.core import ModelBase
from stream_ml.jax.typing import Array

__all__: list[str] = []


@dataclass()
class StreamModel(ModelBase):
    """Stream Model."""

    n_features: int

    # ========================================================================
    # ML

    # TODO: keep moving up the hierarchy!
    def _forward_prior(self, out: Array, data: Data[Array]) -> Array:
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
        for bnd in self.param_bounds.flatvalues():
            out = bnd(out, data, self)
        return out
