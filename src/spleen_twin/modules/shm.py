"""Somatic hypermutation — the AID-driven diversification engine.

Mutation targeting uses the **S5F** model: an empirical 5-mer context model in
which every nucleotide's mutability depends on the two bases either side of
it, and the identity of the replacing base is drawn from a per-5-mer
substitution distribution.

    Yaari G, Vander Heiden JA, Uduman M, Gadala-Maria D, Gupta N,
    Stern JNA, O'Connor KC, Hafler DA, Laserson U, Vigneault F, Kleinstein SH.
    "Models of somatic hypermutation targeting and substitution from
    analyses of high-throughput sequencing data."
    Front Immunol 4:358 (2013).

The shipped tables are the human heavy-chain model ``HH_S5F`` (1024 5-mers),
inferred from 806,860 silent mutations, extracted from the ``shazam`` R
package.

Targeting versus selection
--------------------------
S5F describes where AID *lesions* land, which is a property of sequence
context alone.  The familiar excess of mutations in CDRs that is seen in real
repertoires is largely produced downstream, by selection acting on the
resulting amino-acid changes — it is not an extra targeting bias.  The engine
therefore applies S5F unmodified by default; ``cdr_targeting_boost`` exists
for sensitivity analysis and should be left at 1.0 for mechanistic runs.

Rate
----
``rate_per_base_per_division`` defaults to 1e-3, the canonical SHM rate for a
germinal-centre B cell (roughly one mutation per V region per one to two
divisions).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

#: Canonical germinal-centre SHM rate (per base per cell division).
DEFAULT_SHM_RATE = 1e-3

_BASES = ("A", "C", "G", "T")
_BASE_INDEX = {b: i for i, b in enumerate(_BASES)}


@dataclass(frozen=True)
class Mutation:
    """One nucleotide substitution."""

    position: int
    from_base: str
    to_base: str
    region: str
    silent: bool
    from_aa: str
    to_aa: str

    @property
    def replacement(self) -> bool:
        return not self.silent

    def label(self) -> str:
        return f"{self.from_base}{self.position + 1}{self.to_base}"


class S5FModel:
    """The HH_S5F targeting and substitution model."""

    K = 5
    FLANK = 2

    def __init__(self, mutability_csv: Optional[str] = None,
                 substitution_csv: Optional[str] = None) -> None:
        mut_path = mutability_csv or os.path.join(_DATA, "s5f_mutability.csv")
        sub_path = substitution_csv or os.path.join(_DATA, "s5f_substitution.csv")

        self.mutability: dict[str, float] = {}
        with open(mut_path) as fh:
            for row in csv.DictReader(fh):
                self.mutability[row["fivemer"]] = float(row["mutability"])

        self.substitution: dict[str, np.ndarray] = {}
        with open(sub_path) as fh:
            for row in csv.DictReader(fh):
                fm = row["fivemer"]
                vec = self.substitution.setdefault(fm, np.zeros(4))
                vec[_BASE_INDEX[row["to_base"]]] = float(row["prob"])

        self.mean_mutability = float(np.mean(list(self.mutability.values())))

    def mutability_profile(self, seq: str) -> np.ndarray:
        """Per-position mutability for a nucleotide sequence.

        The two terminal positions at each end lack full 5-mer context and are
        assigned the model's mean mutability.
        """
        seq = seq.upper()
        n = len(seq)
        prof = np.full(n, self.mean_mutability, dtype=float)
        for i in range(self.FLANK, n - self.FLANK):
            fm = seq[i - self.FLANK: i + self.FLANK + 1]
            val = self.mutability.get(fm)
            if val is not None:
                prof[i] = val
        return prof

    def draw_substitution(self, seq: str, pos: int, rng: np.random.Generator) -> str:
        """Draw the replacement base at ``pos`` from the 5-mer substitution model."""
        seq = seq.upper()
        current = seq[pos]
        fm = None
        if self.FLANK <= pos < len(seq) - self.FLANK:
            fm = seq[pos - self.FLANK: pos + self.FLANK + 1]
        probs = self.substitution.get(fm) if fm else None
        if probs is None or probs.sum() <= 0:
            # Uniform over the three alternatives when context is unavailable.
            alts = [b for b in _BASES if b != current]
            return str(rng.choice(alts))
        p = probs.copy()
        if current in _BASE_INDEX:
            p[_BASE_INDEX[current]] = 0.0
        total = p.sum()
        if total <= 0:
            alts = [b for b in _BASES if b != current]
            return str(rng.choice(alts))
        return _BASES[int(rng.choice(4, p=p / total))]


class HypermutationEngine:
    """Applies AID-driven mutations to a B-cell receptor sequence."""

    def __init__(
        self,
        model: Optional[S5FModel] = None,
        rate_per_base_per_division: float = DEFAULT_SHM_RATE,
        cdr_targeting_boost: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.model = model or S5FModel()
        self.rate = rate_per_base_per_division
        self.cdr_targeting_boost = cdr_targeting_boost
        self.rng = rng if rng is not None else np.random.default_rng()

    def expected_mutations(self, seq_len: int, divisions: int = 1) -> float:
        return self.rate * seq_len * divisions

    def mutate(
        self,
        seq_nt: str,
        regions: Optional[Sequence[str]] = None,
        divisions: int = 1,
        translate_fn=None,
    ) -> tuple[str, list[Mutation]]:
        """Introduce Poisson-distributed mutations for ``divisions`` divisions.

        Returns the mutated sequence and the list of mutations applied.
        """
        from .vdj import translate as _default_translate

        translate_fn = translate_fn or _default_translate
        seq = seq_nt.upper()
        n_expected = self.expected_mutations(len(seq), divisions)
        n_mut = int(self.rng.poisson(n_expected))
        if n_mut == 0:
            return seq, []

        prof = self.model.mutability_profile(seq)
        if regions is not None and self.cdr_targeting_boost != 1.0:
            from .regions import CDR_REGIONS
            boost = np.array(
                [self.cdr_targeting_boost if r in CDR_REGIONS else 1.0 for r in regions],
                dtype=float,
            )
            prof = prof * boost[: len(prof)]

        total = prof.sum()
        if total <= 0:
            return seq, []
        p = prof / total
        n_mut = min(n_mut, len(seq))
        positions = self.rng.choice(len(seq), size=n_mut, replace=False, p=p)

        chars = list(seq)
        mutations: list[Mutation] = []
        for pos in sorted(int(x) for x in positions):
            old = chars[pos]
            if old not in _BASE_INDEX:
                continue
            new = self.model.draw_substitution("".join(chars), pos, self.rng)
            codon_start = (pos // 3) * 3
            old_codon = "".join(chars[codon_start: codon_start + 3])
            chars[pos] = new
            new_codon = "".join(chars[codon_start: codon_start + 3])
            if len(old_codon) == 3:
                from_aa = translate_fn(old_codon)
                to_aa = translate_fn(new_codon)
            else:
                from_aa = to_aa = ""
            mutations.append(
                Mutation(
                    position=pos,
                    from_base=old,
                    to_base=new,
                    region=regions[pos] if regions is not None and pos < len(regions) else "unknown",
                    silent=from_aa == to_aa,
                    from_aa=from_aa,
                    to_aa=to_aa,
                )
            )
        return "".join(chars), mutations


__all__ = [
    "DEFAULT_SHM_RATE",
    "Mutation",
    "S5FModel",
    "HypermutationEngine",
]
