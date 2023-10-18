"""Built-in background models."""

from __future__ import annotations

__all__: tuple[str, ...] = ()

from dataclasses import KW_ONLY, dataclass
from math import log
from typing import TYPE_CHECKING, Final

from stream_ml.core._api import SupportsXP
from stream_ml.core.params import set_param
from stream_ml.core.typing import Array

if TYPE_CHECKING:
    from stream_ml.core.typing import ArrayNamespace


# =============================================================================

_five_over_log10: Final = 5 / log(10)


@dataclass(frozen=True)
class Parallax2DistMod(SupportsXP[Array]):
    astrometric_coord: str
    photometric_coord: str

    _: KW_ONLY
    neg_clip_mu: float = 1e-30
    array_namespace: ArrayNamespace[Array]

    def __call__(
        self, pars: dict[str, Array | dict[str, Array]], /
    ) -> dict[str, Array | dict[str, Array]]:
        # Convert parallax (mas) to distance modulus
        # .. math::
        #       distmod = 5 log10(d [pc]) - 5 = -5 log10(plx [arcsec]) - 5
        #               = -5 log10(plx [mas] / 1e3) - 5
        #               = 10 - 5 log10(plx [mas])
        # dm = 10 - 5 * xp.log10(pars["photometric.parallax"]["mu"].reshape((-1, 1)))
        mu = self.xp.clip(pars[self.astrometric_coord]["mu"], self.neg_clip_mu)  # type: ignore[arg-type]
        dm = 10 - 5 * self.xp.log10(mu)
        ln_dm_sigma = self.xp.log(
            _five_over_log10
            * self.xp.exp(pars[self.astrometric_coord]["ln-sigma"])  # type: ignore[arg-type]
            / mu
        )

        # Set the distance modulus
        set_param(pars, (self.photometric_coord, "mu"), dm)
        set_param(pars, (self.photometric_coord, "ln-sigma"), ln_dm_sigma)

        return pars
