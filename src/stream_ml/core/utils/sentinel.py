"""Stream Memberships Likelihood, with ML."""

# STDLIB
from enum import Enum, unique


@unique
class Sentinel(Enum):
    """Sentinel codes."""

    MISSING = "missing"


MISSING = Sentinel.MISSING
