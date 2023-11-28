"""Stream Memberships Likelihood, with ML."""

from enum import Enum, unique
from typing import Literal, TypeAlias


@unique
class Sentinel(Enum):
    """Sentinel codes."""

    MISSING = "missing"


MISSING = Sentinel.MISSING


MissingT: TypeAlias = Literal[Sentinel.MISSING]
