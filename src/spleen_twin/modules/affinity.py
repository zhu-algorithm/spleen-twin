"""Affinity scoring — the selection pressure driving the germinal centre.

.. warning::

   :class:`SurrogateAffinityOracle` is a **fitness function, not a binding
   predictor.**  It produces a smooth, epistatic, saturating landscape with
   the right dynamic range so that affinity maturation behaves like the real
   process; it makes no claim to predict the dissociation constant of any
   real antibody-antigen pair.  Every number it returns should be read as
   "simulated KD".  :class:`AffinityOracle` is the interface to replace it
   with a real structural predictor.

Why a surrogate at all
----------------------
Nothing available today predicts antibody-antigen affinity from sequence at
the accuracy a maturation simulation would need.  The AIntibody blind
benchmark is the cautionary datum: across 29 groups and 511 AI-designed
antibodies, only 13% achieved a 20-fold affinity improvement, and a non-AI
consensus baseline outperformed all but two entries.  Pretending otherwise
inside a simulator would quietly launder a guess into a result.  So the
surrogate is explicit, cheap, and clearly labelled, and the oracle interface
lets a structural method be swapped in for the handful of clones where it is
worth the compute.

Construction
------------
The paratope is taken as the concatenated CDR loops (CDR-H3 weighted most
heavily, reflecting its dominance of the binding surface).  Complementarity
to the epitope is scored over a sliding register from published
amino-acid property scales:

* **hydrophobic burial** — Janin interior-to-surface transfer free energy
  (Janin J, Nature 277:491-492, 1979), via Biopython ``ProtParamData.ja``
* **electrostatic complementarity** — formal side-chain charge at pH 7.4
* **steric fit** — residue volume complementarity from residue molecular
  weights (Biopython ``IUPACData.protein_weights``)
* **polar contact potential** — side-chain hydrogen-bond donor and acceptor
  counts

Scores are mapped onto a log10 KD scale calibrated so that an unmutated
naive repertoire spans roughly 1e-5 to 1e-7 M and the attainable ceiling is
about 1e-11 M, matching the range quoted for real repertoires (naive
1e-7..1e-8 M; matured 1e-9..1e-11 M; best-in-class therapeutics
1e-10..1e-12 M).

A folding-stability penalty makes framework replacement mutations costly,
so the landscape cannot be climbed by destroying the domain — the same
coupling that the developability module later scores explicitly.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from Bio.Data import IUPACData
from Bio.SeqUtils import ProtParamData

#: Gas constant times 310 K, in kcal/mol — binding free energies are reported
#: at physiological temperature.
RT_KCAL = 1.9872e-3 * 310.0

#: Formal side-chain charge at pH 7.4.
CHARGE = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
}

#: Side-chain hydrogen-bond donors and acceptors.
HB_DONORS = {"K": 3, "R": 3, "W": 1, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1, "C": 1}
HB_ACCEPTORS = {"D": 2, "E": 2, "N": 1, "Q": 1, "H": 1, "S": 1, "T": 1, "Y": 1}

_JANIN = ProtParamData.ja
_WEIGHTS = IUPACData.protein_weights


def _volume(aa: str) -> float:
    """Residue volume proxy: molecular weight minus one water."""
    return _WEIGHTS.get(aa, 110.0) - 18.0


@dataclass
class AffinityResult:
    """Outcome of scoring one antibody against one epitope."""

    kd_molar: float
    log10_kd: float
    dg_kcal: float
    raw_score: float
    stability_penalty: float
    register: int
    epitope: str

    @property
    def pkd(self) -> float:
        return -self.log10_kd


class AffinityOracle(ABC):
    """Interface for anything that scores an antibody against an epitope."""

    @abstractmethod
    def score(self, paratope: dict[str, str], epitope_seq: str,
              framework_replacements: int = 0) -> AffinityResult:
        """Return the simulated affinity of one antibody for one epitope."""

    def kd(self, *args, **kwargs) -> float:
        return self.score(*args, **kwargs).kd_molar


class SurrogateAffinityOracle(AffinityOracle):
    """Transparent physicochemical complementarity surrogate.

    Parameters
    ----------
    cdr_weights
        Relative contribution of each CDR loop to the paratope.  The default
        reflects the dominance of CDR-H3 in antigen contact.
    ceiling_log10_kd
        Best attainable simulated affinity; the landscape saturates here.
    floor_log10_kd
        Affinity of a paratope with no complementarity at all.
    stability_cost_per_fr_mutation
        Free-energy cost (kcal/mol) charged per framework replacement
        mutation, representing loss of folding stability.
    """

    def __init__(
        self,
        cdr_weights: Optional[dict[str, float]] = None,
        ceiling_log10_kd: float = -11.0,
        floor_log10_kd: float = -4.5,
        stability_cost_per_fr_mutation: float = 0.18,
        hydrophobic_weight: float = 1.0,
        electrostatic_weight: float = 1.4,
        steric_weight: float = 0.6,
        hbond_weight: float = 0.5,
    ) -> None:
        self.cdr_weights = cdr_weights or {"cdr1": 0.6, "cdr2": 0.7, "cdr3": 1.8}
        self.ceiling = ceiling_log10_kd
        self.floor = floor_log10_kd
        self.stability_cost = stability_cost_per_fr_mutation
        self.w_hyd = hydrophobic_weight
        self.w_ele = electrostatic_weight
        self.w_ste = steric_weight
        self.w_hb = hbond_weight
        self._calibration = None   # set by calibrate()
        self._E = self._build_energy_matrix()
        self._score_cache: dict[tuple, tuple[float, int]] = {}

    # -- precomputed pair-energy table -------------------------------------
    #: Amino-acid alphabet used to index the pair-energy matrix.
    ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def _build_energy_matrix(self) -> np.ndarray:
        """Tabulate the pair energy for all 400 residue pairs once."""
        idx = {a: i for i, a in enumerate(self.ALPHABET)}
        E = np.zeros((len(self.ALPHABET), len(self.ALPHABET)), dtype=float)
        for a, i in idx.items():
            for b, j in idx.items():
                E[i, j] = self._pair_energy(a, b)
        return E

    def _encode(self, seq: str) -> np.ndarray:
        idx = {a: i for i, a in enumerate(self.ALPHABET)}
        return np.array([idx.get(c, 0) for c in seq], dtype=np.intp)

    # -- per-pair complementarity ------------------------------------------
    def _pair_energy(self, a: str, b: str) -> float:
        """Favourable (negative) energy for a paratope/epitope residue pair."""
        ja_a, ja_b = _JANIN.get(a, 0.0), _JANIN.get(b, 0.0)
        # Janin is positive for buried/hydrophobic residues; two hydrophobes
        # in contact bury surface and gain free energy.
        hyd = -self.w_hyd * max(0.0, ja_a) * max(0.0, ja_b)

        qa, qb = CHARGE.get(a, 0.0), CHARGE.get(b, 0.0)
        ele = self.w_ele * qa * qb          # opposite charges give negative
        if qa * qb > 0:
            ele *= 1.5                      # like charges penalised harder

        # Steric complementarity: large against small packs better than
        # large against large.
        va, vb = _volume(a), _volume(b)
        ste = self.w_ste * ((va + vb) - 230.0) ** 2 / 2.0e4

        hb = -self.w_hb * 0.5 * (
            min(HB_DONORS.get(a, 0), HB_ACCEPTORS.get(b, 0))
            + min(HB_DONORS.get(b, 0), HB_ACCEPTORS.get(a, 0))
        )
        return hyd + ele + ste + hb

    def _best_register(self, paratope_seq: str, weights: np.ndarray,
                       epitope_seq: str) -> tuple[float, int]:
        """Slide the paratope across the epitope and keep the best contact set.

        Scores every register at once: ``M[i, j]`` is the weighted energy of
        paratope residue *i* against epitope residue *j*, and the score for
        offset *o* is the sum of the diagonal ``j - i == o``.
        """
        le, lp = len(epitope_seq), len(paratope_seq)
        if le == 0 or lp == 0:
            return 0.0, 0
        pi = self._encode(paratope_seq)
        ei = self._encode(epitope_seq)
        M = self._E[np.ix_(pi, ei)] * weights[:, None]
        offsets = np.arange(-(lp - 1), le)
        sums = np.array([np.trace(M, offset=int(o)) for o in offsets])
        k = int(np.argmin(sums))
        best = float(sums[k])
        return (best, int(offsets[k])) if best < 0 else (0.0, 0)

    # -- public API ---------------------------------------------------------
    def raw_score(self, paratope: dict[str, str], epitope_seq: str) -> tuple[float, int]:
        key = (paratope.get("cdr1", ""), paratope.get("cdr2", ""),
               paratope.get("cdr3", ""), epitope_seq)
        hit = self._score_cache.get(key)
        if hit is not None:
            return hit

        seq_parts, weight_parts = [], []
        for loop, w in self.cdr_weights.items():
            s = paratope.get(loop, "")
            if not s:
                continue
            seq_parts.append(s)
            weight_parts.append(np.full(len(s), w, dtype=float))
        if not seq_parts:
            return 0.0, 0
        out = self._best_register("".join(seq_parts), np.concatenate(weight_parts), epitope_seq)
        if len(self._score_cache) < 2_000_000:
            self._score_cache[key] = out
        return out

    def calibrate(
        self,
        naive_scores: Sequence[float],
        naive_median_log10_kd: float = -6.3,
        best_naive_log10_kd: float = -8.0,
        best_percentile: float = 0.1,
    ) -> None:
        """Anchor the score-to-KD map on an unmutated naive repertoire.

        Two anchors are used so the mapping reproduces both the typical and
        the best-of-repertoire naive affinity reported for real responses:
        the median naive clone sits at ``naive_median_log10_kd`` and the
        ``best_percentile`` clone at ``best_naive_log10_kd``.  Everything
        beyond that has to be earned by somatic hypermutation, which is what
        makes maturation in this model a genuine climb rather than a lottery
        already won in the naive repertoire.
        """
        arr = np.asarray(list(naive_scores), dtype=float)
        med = float(np.median(arr))
        unit = float(np.percentile(arr, best_percentile) - med)
        if unit == 0:
            unit = -1.0
        span = naive_median_log10_kd - self.ceiling          # > 0
        gain_at_best = naive_median_log10_kd - best_naive_log10_kd
        frac = min(0.95, max(1e-3, gain_at_best / span))
        k = -math.log(1.0 - frac)                            # so z=1 gives gain_at_best
        self._calibration = (med, unit, naive_median_log10_kd, k)

    def score(self, paratope: dict[str, str], epitope_seq: str,
              framework_replacements: int = 0) -> AffinityResult:
        raw, register = self.raw_score(paratope, epitope_seq)

        if self._calibration is None:
            med, unit, anchor, k = 0.0, -1.0, -6.3, 0.45
        else:
            med, unit, anchor, k = self._calibration

        # Position relative to the naive median, in units of "best naive clone".
        z = (raw - med) / unit if unit else 0.0
        # Saturating climb toward the ceiling: each unit of z buys less.
        span = anchor - self.ceiling                   # positive magnitude
        improvement = span * (1.0 - math.exp(-k * max(0.0, z)))
        decline = (self.floor - anchor) * (1.0 - math.exp(-k * max(0.0, -z)))
        log10_kd = anchor - improvement + decline

        penalty = self.stability_cost * max(0, framework_replacements)
        log10_kd += penalty / (RT_KCAL * math.log(10))

        log10_kd = float(min(self.floor, max(self.ceiling, log10_kd)))
        dg = RT_KCAL * math.log(10) * log10_kd
        return AffinityResult(
            kd_molar=10.0 ** log10_kd,
            log10_kd=log10_kd,
            dg_kcal=dg,
            raw_score=raw,
            stability_penalty=penalty,
            register=register,
            epitope=epitope_seq,
        )


def paratope_from(region_map, seq_aa: str) -> dict[str, str]:
    """Extract CDR loop amino-acid sequences using an IMGT region map."""
    out: dict[str, list[str]] = {"cdr1": [], "cdr2": [], "cdr3": []}
    for i, aa in enumerate(seq_aa):
        lab = region_map.region_of(i * 3)
        if lab in out:
            out[lab].append(aa)
    return {k: "".join(v) for k, v in out.items()}


__all__ = [
    "AffinityOracle",
    "SurrogateAffinityOracle",
    "AffinityResult",
    "paratope_from",
    "RT_KCAL",
    "CHARGE",
]
