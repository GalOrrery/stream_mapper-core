"""Package Setup."""

__all__: tuple[str, ...] = ()

from itertools import chain
from typing import Final

from mypy_extensions import trait

from stream_mapper.core._data import Data

WEIGHT_NAME: Final = "ln-weight"
BACKGROUND_KEY: Final = "background"
PACK_PARAM_JOIN: Final = "_"
IS_COMPILED = hasattr(Data, "__mypyc_attrs__")


@trait
class CompiledShim:
    """Shim to make the compiled code work with the uncompiled version."""

    def _mypyc_init_descriptor(self) -> None:
        """Initialize the descriptor."""
        if not IS_COMPILED:
            return

        # This is only necessary when compiled with mypyc.
        # I think there's a bug in mypyc that causes it to not
        # call the __set__ method of the descriptor and instead
        # just set the attribute directly.
        for k, v in chain(*(c.__dict__.items() for c in self.__class__.mro())):
            # Check if the attribute is a descriptor and if it's
            # not already set.
            if (
                k not in getattr(self, "__dataclass_fields__", ())
                or not hasattr(v, "__set__")
                or hasattr(self, "_" + k)
            ):
                continue

            v.__set__(self, getattr(self, k))
            object.__setattr__(self, k, getattr(self, "_" + k))
