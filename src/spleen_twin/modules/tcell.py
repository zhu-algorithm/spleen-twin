"""CD4+ T-cell priming and Tfh differentiation.

Follicular helper T cells are the limiting resource of the germinal centre.
Their number sets how many B cells can be rescued per selection round, and
therefore how hard the competition is.  This module turns an antigen
presentation event into a Tfh population that the germinal centre draws on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TfhCompartment:
    """The help-providing population available to a germinal centre."""

    n_tfh: int
    help_per_tfh: float
    quality: float          # 0-1, how strong the priming was

    @property
    def total_help(self) -> float:
        return self.n_tfh * self.help_per_tfh


class TCellCompartment:
    """Primes CD4+ T cells in the T-cell zone and makes Tfh."""

    def __init__(
        self,
        naive_cd4_pool: int = 200_000,
        precursor_frequency: float = 1e-5,
        expansion_factor: float = 1_000.0,
        tfh_fraction: float = 0.12,
        help_per_tfh: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.naive_cd4_pool = naive_cd4_pool
        self.precursor_frequency = precursor_frequency
        self.expansion_factor = expansion_factor
        self.tfh_fraction = tfh_fraction
        self.help_per_tfh = help_per_tfh
        self.rng = rng if rng is not None else np.random.default_rng()

    def prime(self, presentation) -> TfhCompartment:
        """Expand antigen-specific CD4+ T cells and differentiate Tfh."""
        precursors = self.naive_cd4_pool * self.precursor_frequency
        quality = float(np.clip(presentation.mhcii_capacity, 0.0, 1.0))
        expanded = precursors * self.expansion_factor * (0.25 + 0.75 * quality)
        n_tfh = int(self.rng.poisson(max(0.0, expanded * self.tfh_fraction)))
        return TfhCompartment(
            n_tfh=n_tfh,
            help_per_tfh=self.help_per_tfh * (0.5 + 0.5 * quality),
            quality=quality,
        )


__all__ = ["TCellCompartment", "TfhCompartment"]
