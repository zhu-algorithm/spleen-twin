"""Antigen input and epitope mapping.

The antigen is the entry point of the whole organ model: it is taken up by
antigen-presenting cells, processed into peptides for MHC-II display to CD4+
T cells, and held intact on follicular dendritic cells as the target that
germinal-centre B cells compete for.

The default antigen is hen egg lysozyme (UniProt P00698), the classical model
antigen of germinal-centre biology and the target of the HyHEL antibody
series.  Any sequence can be substituted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Signal peptide length of the P00698 precursor; mature lysozyme is 1-129
#: in the conventional numbering used by the structural literature.
HEL_SIGNAL_PEPTIDE_LEN = 18


def read_fasta(path: str) -> tuple[str, str]:
    header, parts = "", []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            elif line:
                parts.append(line)
    return header, "".join(parts)


@dataclass
class Epitope:
    """A contiguous stretch of antigen surface targeted by selection.

    ``start``/``end`` are 1-based inclusive positions in the mature antigen.
    """

    name: str
    start: int
    end: int
    sequence: str
    conformational: bool = False
    notes: str = ""

    def __len__(self) -> int:
        return len(self.sequence)


@dataclass
class Antigen:
    """An immunogen with one or more mapped epitopes."""

    name: str
    sequence: str                       # mature chain used for epitope numbering
    source: str = ""
    precursor: Optional[str] = None
    epitopes: list[Epitope] = field(default_factory=list)
    dose_nm: float = 10.0               # molar dose presented on FDCs

    def add_epitope(self, name: str, start: int, end: int, **kw) -> Epitope:
        ep = Epitope(name=name, start=start, end=end,
                     sequence=self.sequence[start - 1: end], **kw)
        self.epitopes.append(ep)
        return ep

    def immunodominant(self) -> Epitope:
        if not self.epitopes:
            raise ValueError(f"antigen {self.name} has no mapped epitopes")
        return self.epitopes[0]

    def peptides(self, length: int = 15, step: int = 3) -> list[tuple[int, str]]:
        """Overlapping peptides, the substrate for MHC-II presentation."""
        return [
            (i + 1, self.sequence[i: i + length])
            for i in range(0, max(1, len(self.sequence) - length + 1), step)
        ]


def hen_egg_lysozyme(dose_nm: float = 10.0) -> Antigen:
    """Load hen egg lysozyme (UniProt P00698) with its canonical epitopes.

    The precursor in UniProt carries an 18-residue signal peptide; the mature
    129-residue chain is used so that epitope positions follow the numbering
    of the HyHEL structural literature.
    """
    header, precursor = read_fasta(os.path.join(_DATA, "P00698_HEL.fasta"))
    mature = precursor[HEL_SIGNAL_PEPTIDE_LEN:]
    ag = Antigen(
        name="HEL",
        sequence=mature,
        source=header,
        precursor=precursor,
        dose_nm=dose_nm,
    )
    # A contiguous surface stretch spanning the HyHEL-5 contact region.
    ag.add_epitope(
        "HyHEL5_region", 41, 60,
        notes="contiguous surface patch overlapping the HyHEL-5 footprint "
              "(R45, N46, T47, D48, G49, S50, T51, Q53)",
    )
    # The HyHEL-10 footprint, used as a second, independent target.
    ag.add_epitope(
        "HyHEL10_region", 88, 102,
        notes="contiguous stretch overlapping the HyHEL-10 footprint "
              "(T89, N93, K96, K97, I98, S100, D101, G102)",
    )
    return ag


__all__ = ["Antigen", "Epitope", "hen_egg_lysozyme", "read_fasta",
           "HEL_SIGNAL_PEPTIDE_LEN"]
