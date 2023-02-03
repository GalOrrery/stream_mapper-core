"""Package Setup."""

from itertools import chain

from mypy_extensions import trait

from stream_ml.core.data import Data

IS_COMPILED = hasattr(Data, "__mypyc_attrs__")


@trait
class CompiledShim:
    """Shim to make the compiled code work with the uncompiled version."""

    def _init_descriptor(self) -> None:
        """Initialize the descriptor."""
        if not IS_COMPILED:
            return

        # This is only necessary when compiled with mypyc.
        # I think there's a bug in mypyc that causes it to not
        # call the __set__ method of the descriptor and instead
        # just set the attribute directly.
        for k, v in chain(*(c.__dict__.items() for c in self.__class__.mro())):
            if (
                k not in getattr(self, "__dataclass_fields__", ())
                or not hasattr(v, "__set__")
                or hasattr(self, "_" + k)
            ):
                continue

            v.__set__(self, getattr(self, k))
            object.__setattr__(self, k, getattr(self, "_" + k))
