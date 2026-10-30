"""Antigen presentation — uptake, processing and display.

Two presentation routes matter for a humoral response and both are modelled:

* **MHC-II peptide presentation** by dendritic cells and macrophages, which
  licenses CD4+ T cells and ultimately sets how much T-cell help the
  germinal centre can supply.
* **Native antigen display on follicular dendritic cells** as immune
  complexes.  FDCs do not process antigen; they hold it intact, which is
  what germinal-centre B cells compete to capture with their BCR.

MHC-II binding is scored with a transparent heuristic over published
amino-acid property scales, not a trained predictor.  It ranks peptides of
one antigen against each other, which is all the organ model needs; it is
not an epitope-prediction tool and should not be used as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from Bio.SeqUtils import ProtParamData

#: MHC-II binding groove accommodates a 9-residue core with anchors at
#: P1, P4, P6 and P9.
MHCII_CORE = 9
ANCHOR_POSITIONS = (0, 3, 5, 8)

#: P1 of the MHC-II groove is a deep hydrophobic pocket.
_P1_PREFERRED = set("FWYLIVM")
_KD = ProtParamData.kd


@dataclass
class PresentedPeptide:
    start: int
    peptide: str
    core: str
    score: float


@dataclass
class AntigenPresentation:
    """What the APC compartment produced from one antigen dose."""

    peptides: list[PresentedPeptide]
    fdc_antigen_nm: float
    mhcii_capacity: float

    def top(self, n: int = 10) -> list[PresentedPeptide]:
        return sorted(self.peptides, key=lambda p: -p.score)[:n]


class AntigenPresentingCompartment:
    """Dendritic cells, macrophages and follicular dendritic cells."""

    def __init__(
        self,
        uptake_efficiency: float = 0.35,
        fdc_retention: float = 0.55,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.uptake_efficiency = uptake_efficiency
        self.fdc_retention = fdc_retention
        self.rng = rng if rng is not None else np.random.default_rng()

    def _core_score(self, core: str) -> float:
        """Heuristic groove-fit score for a 9-mer core."""
        if len(core) < MHCII_CORE:
            return -np.inf
        s = 0.0
        for i in ANCHOR_POSITIONS:
            aa = core[i]
            s += _KD.get(aa, 0.0) * (1.6 if i == 0 and aa in _P1_PREFERRED else 1.0)
        # Prolines in the core disrupt the extended backbone conformation.
        s -= 2.0 * core[1:8].count("P")
        return s

    def present(self, antigen, peptide_length: int = 15) -> AntigenPresentation:
        """Process an antigen into MHC-II peptides and load FDCs."""
        presented: list[PresentedPeptide] = []
        for start, pep in antigen.peptides(length=peptide_length, step=1):
            best, best_core = -np.inf, ""
            for i in range(0, len(pep) - MHCII_CORE + 1):
                core = pep[i:i + MHCII_CORE]
                sc = self._core_score(core)
                if sc > best:
                    best, best_core = sc, core
            if best > -np.inf:
                presented.append(PresentedPeptide(start, pep, best_core, float(best)))

        fdc = antigen.dose_nm * self.uptake_efficiency * self.fdc_retention
        scores = np.array([p.score for p in presented], dtype=float)
        # Help capacity scales with how well the best peptides load MHC-II.
        capacity = float(np.tanh(max(0.0, np.percentile(scores, 95)) / 12.0)) if len(scores) else 0.0
        return AntigenPresentation(presented, fdc_antigen_nm=fdc, mhcii_capacity=capacity)


__all__ = ["AntigenPresentingCompartment", "AntigenPresentation", "PresentedPeptide",
           "MHCII_CORE", "ANCHOR_POSITIONS"]
