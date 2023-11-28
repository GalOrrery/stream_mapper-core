"""Built-in background models."""

from __future__ import annotations

__all__: tuple[str, ...] = ()


class WhereRequiredError(ValueError):
    """Raised when a model requires the `where` argument."""

    def __init__(self) -> None:
        super().__init__("this model requires the `where` argument")
