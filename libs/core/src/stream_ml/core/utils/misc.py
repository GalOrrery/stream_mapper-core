"""Core feature."""

from __future__ import annotations

# STDLIB
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # THIRD-PARTY

    # LOCAL
    from stream_ml.core._typing import ArrayT, ParsT

__all__: list[str] = []


def get_params_for_model(
    name: str | tuple[str, ...], pars: ParsT[ArrayT]
) -> ParsT[ArrayT]:
    """Get parameters for model.

    Parameters
    ----------
    name : str | tuple[str, ...]
        The name of the model.
    pars : ParsT
        Parameters from which to get the sub-parameters

    Returns
    -------
    ParsT
    """
    n = "_".join((name,) if isinstance(name, str) else name) + "_"
    lenn = len(n)

    return {k[lenn:]: v for k, v in pars.items() if k.startswith(n)}
