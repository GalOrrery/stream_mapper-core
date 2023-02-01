"""Stream Memberships Likelihood, with ML."""

# THIRD-PARTY
import jax as _jax

_jax.config.update("jax_array", True)
_jax.config.update("jax_enable_x64", True)
# isort: split

# LOCAL
from stream_ml.core.data import Data  # noqa: E402
from stream_ml.jax import background, stream, utils  # noqa: E402
from stream_ml.jax.mixture import MixtureModel  # noqa: E402

__all__ = [
    # modules
    "background",
    "stream",
    "utils",
    # classes
    "MixtureModel",
    "Data",
]
