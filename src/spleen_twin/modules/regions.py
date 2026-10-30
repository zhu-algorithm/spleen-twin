"""IMGT region annotation for rearranged heavy chains.

Maps every nucleotide of a V-REGION onto its IMGT framework/CDR label so that
somatic hypermutation and selection can be scored per region.

Two reference tables are used, both shipped with the package:

* ``imgt_v_regions.csv`` — the IMGT unique numbering boundary vector for the
  312-nt gapped V segment (FR1 aa 1-26, CDR1 27-38, FR2 39-55, CDR2 56-65,
  FR3 66-104), extracted from shazam's ``IMGT_V_BY_REGIONS``.

      Lefranc MP, et al. "IMGT unique numbering for immunoglobulin and T cell
      receptor variable domains and Ig superfamily V-like domains."
      Dev Comp Immunol 27:55-77 (2003).

* ``IMGT_gapped_IGHV.fasta`` — 344 IMGT-gapped human IGHV alleles, extracted
  from the ``tigger`` R package's ``GermlineIGHV`` reference.

An OLGA germline V segment is ungapped, so the gap columns of the matching
IMGT allele are transferred onto it.  When an allele is absent from the gapped
reference, the gap pattern of another allele of the same gene is used (alleles
of one gene differ by substitutions, essentially never by indels in the
framework).  Two rare OLGA alleles have no usable donor and are reported as
unannotated rather than guessed at.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Length of the IMGT-gapped V frame covered by the boundary table.
IMGT_V_FRAME_NT = 312

REGION_ORDER = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")
CDR_REGIONS = frozenset({"cdr1", "cdr2", "cdr3"})


def _load_boundaries() -> list[str]:
    with open(os.path.join(_DATA, "imgt_v_regions.csv")) as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["nt_position"]))
    return [r["region"] for r in rows]


def _load_gapped() -> dict[str, str]:
    seqs: dict[str, str] = {}
    name = None
    with open(os.path.join(_DATA, "IMGT_gapped_IGHV.fasta")) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                name = line[1:]
                seqs[name] = ""
            elif name:
                seqs[name] += line.upper()
    return seqs


@dataclass
class RegionMap:
    """Per-nucleotide region labels for one rearranged chain."""

    labels: list[str]          # one label per nucleotide of seq_nt
    v_allele: str
    donor_allele: str          # allele whose gap pattern was used
    annotated: bool            # False when no IMGT donor was available

    def mask(self, region: str) -> list[int]:
        return [i for i, lab in enumerate(self.labels) if lab == region]

    def cdr_positions(self) -> list[int]:
        return [i for i, lab in enumerate(self.labels) if lab in CDR_REGIONS]

    def region_of(self, i: int) -> str:
        return self.labels[i] if 0 <= i < len(self.labels) else "unknown"

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for lab in self.labels:
            out[lab] = out.get(lab, 0) + 1
        return out


class ImgtAnnotator:
    """Assigns IMGT region labels to rearranged heavy chains."""

    def __init__(self) -> None:
        self.boundaries = _load_boundaries()
        self.gapped = _load_gapped()
        self.by_gene: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name, seq in self.gapped.items():
            self.by_gene[name.split("*")[0]].append((name, seq))
        self._cache: dict[tuple[str, int], tuple[Optional[str], Optional[list[str]]]] = {}

    def _v_labels(self, v_allele: str, prefix_len: int):
        """Region label per nucleotide of the ungapped V prefix (up to the Cys)."""
        key = (v_allele, prefix_len)
        if key in self._cache:
            return self._cache[key]

        target = prefix_len + 3  # prefix plus the conserved Cys codon
        candidates: list[tuple[str, str]] = []
        if v_allele in self.gapped:
            candidates.append((v_allele, self.gapped[v_allele]))
        candidates += [
            (n, s) for n, s in self.by_gene.get(v_allele.split("*")[0], []) if n != v_allele
        ]

        for donor, gseq in candidates:
            frame = gseq[:IMGT_V_FRAME_NT]
            if len(frame.replace(".", "")) != target:
                continue
            labels = [
                self.boundaries[i] for i, ch in enumerate(frame) if ch != "."
            ]
            # Drop the trailing conserved Cys codon; it belongs to CDR3.
            result = (donor, labels[:prefix_len])
            self._cache[key] = result
            return result

        self._cache[key] = (None, None)
        return None, None

    def annotate(self, rearrangement) -> RegionMap:
        """Label every nucleotide of ``rearrangement.seq_nt``.

        The sequence is laid out as V prefix + CDR3 + FR4; the recombinator
        records the first and last lengths, so only the V prefix requires the
        IMGT lookup.
        """
        v_len = rearrangement.v_prefix_nt
        cdr3_len = len(rearrangement.cdr3_nt)
        fr4_len = rearrangement.fr4_nt

        donor, labels = self._v_labels(rearrangement.v_gene, v_len)
        if labels is None:
            return RegionMap(
                labels=["unknown"] * v_len + ["cdr3"] * cdr3_len + ["fwr4"] * fr4_len,
                v_allele=rearrangement.v_gene,
                donor_allele="",
                annotated=False,
            )
        return RegionMap(
            labels=list(labels) + ["cdr3"] * cdr3_len + ["fwr4"] * fr4_len,
            v_allele=rearrangement.v_gene,
            donor_allele=donor,
            annotated=True,
        )


__all__ = ["ImgtAnnotator", "RegionMap", "REGION_ORDER", "CDR_REGIONS", "IMGT_V_FRAME_NT"]
