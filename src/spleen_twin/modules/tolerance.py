"""Central and peripheral tolerance.

A repertoire is not just what V(D)J can build; it is what survives
self-screening.  Three outcomes are modelled, following the canonical
immature-B-cell checkpoint:

* **deletion** — strongly self-reactive clones die,
* **receptor editing** — the light chain is re-rearranged and the cell gets
  another try,
* **anergy** — weakly self-reactive clones survive but are functionally
  silenced and excluded from germinal centres.

Self-reactivity is scored with the same surrogate oracle used for the
antigen, applied to a panel of self peptides.  This is a mechanism model,
not a prediction of autoreactivity for any real antibody.

Also modelled is the pre-BCR checkpoint, which simply requires a productive
heavy chain — the step that removes the ~2/3 of rearrangements that V(D)J
produces out of frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np


class ToleranceOutcome(str, Enum):
    NAIVE = "naive"
    DELETED = "deleted"
    EDITED = "edited"
    ANERGIC = "anergic"


@dataclass
class ToleranceResult:
    outcome: ToleranceOutcome
    self_reactivity: float
    edits_used: int = 0


import os

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_self_panel(peptide_length: Optional[int] = None, step: int = 10) -> list[str]:
    """Abundant human self proteins, tiled into peptides.

    The panel is built from eight ubiquitous human proteins fetched from
    UniProt (serum albumin P02768, beta-actin P60709, histone H3.1 P68431,
    insulin P01308, haemoglobin alpha P69905, lamin A/C P02545, GAPDH
    P04406, IgG1 heavy constant P01857).  It defines the shape of the self
    landscape that immature B cells are screened against; it is not a
    catalogue of autoantigens.
    """
    seqs, cur = [], []
    path = os.path.join(_DATA, "self_panel.fasta")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))

    if peptide_length is None:
        # Whole proteins: the oracle already slides the paratope across the
        # whole sequence, so tiling adds cost without adding coverage.
        return seqs

    peptides: list[str] = []
    for s in seqs:
        for i in range(0, max(1, len(s) - peptide_length + 1), step):
            peptides.append(s[i:i + peptide_length])
    return peptides


class ToleranceCheckpoint:
    """Screens immature B cells against a self panel."""

    def __init__(
        self,
        oracle,
        self_panel: Sequence[str],
        delete_threshold_log10_kd: float = -8.0,
        anergy_threshold_log10_kd: float = -7.0,
        max_edits: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.oracle = oracle
        self.self_panel = list(self_panel)
        self.delete_threshold = delete_threshold_log10_kd
        self.anergy_threshold = anergy_threshold_log10_kd
        self.max_edits = max_edits
        self.rng = rng if rng is not None else np.random.default_rng()

    def calibrate(
        self,
        sample_paratopes,
        deleted_fraction: float = 0.55,
        anergic_fraction: float = 0.20,
    ) -> dict:
        """Set thresholds from the repertoire's own self-reactivity spread.

        Self-reactivity here is a best-register score against whole self
        proteins, which is an extreme-value statistic: its absolute scale
        depends on how much self sequence is screened, so an absolute KD
        cut-off is not transferable.  Thresholds are therefore placed at
        quantiles chosen to reproduce the measured outcome of human central
        tolerance — roughly 55-75% of early immature B cells are
        self-reactive and are removed or silenced before the mature naive
        stage.

            Wardemann H, Yurasov S, Schaefer A, Young JW, Meffre E,
            Nussenzweig MC. "Predominant autoantibody production by early
            human B cells." Science 301:1374-1377 (2003).
        """
        import numpy as _np

        vals = _np.array([self.self_reactivity(p) for p in sample_paratopes])
        self.delete_threshold = float(_np.quantile(vals, deleted_fraction))
        self.anergy_threshold = float(_np.quantile(vals, deleted_fraction + anergic_fraction))
        return {
            "n_sampled": int(vals.size),
            "delete_threshold_log10_kd": self.delete_threshold,
            "anergy_threshold_log10_kd": self.anergy_threshold,
            "target_deleted_fraction": deleted_fraction,
            "target_anergic_fraction": anergic_fraction,
        }

    def self_reactivity(self, paratope: dict) -> float:
        """Strongest (most negative log10 KD) binding to any self peptide."""
        best = 0.0
        for pep in self.self_panel:
            r = self.oracle.score(paratope, pep)
            best = min(best, r.log10_kd) if best else r.log10_kd
        return best

    def screen(self, paratope: dict, edits_used: int = 0) -> ToleranceResult:
        """Classify one immature B cell.

        ``max_edits`` defaults to 0 because receptor editing re-rearranges
        the *light* chain; in a heavy-chain-only run there is nothing for
        editing to change, so a self-reactive cell is deleted.  Set
        ``max_edits`` above zero only when light chains are being modelled.
        """
        sr = self.self_reactivity(paratope)
        if sr <= self.delete_threshold:
            if edits_used < self.max_edits:
                return ToleranceResult(ToleranceOutcome.EDITED, sr, edits_used + 1)
            return ToleranceResult(ToleranceOutcome.DELETED, sr, edits_used)
        if sr <= self.anergy_threshold:
            return ToleranceResult(ToleranceOutcome.ANERGIC, sr, edits_used)
        return ToleranceResult(ToleranceOutcome.NAIVE, sr, edits_used)


def pre_bcr_checkpoint(rearrangement) -> bool:
    """A pro-B cell survives only with a productive heavy chain."""
    return bool(rearrangement is not None and rearrangement.productive)


__all__ = ["ToleranceCheckpoint", "ToleranceResult", "ToleranceOutcome",
           "pre_bcr_checkpoint", "load_self_panel"]
