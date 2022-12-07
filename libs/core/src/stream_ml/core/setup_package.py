"""Stream Memberships Likelihood, with ML."""

__all__ = ["HAS_TQDM"]

HAS_TQDM: bool

try:
    # THIRD-PARTY
    import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True
