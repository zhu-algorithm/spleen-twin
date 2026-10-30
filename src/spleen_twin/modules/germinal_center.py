"""The germinal centre — iterative mutation and selection.

A germinal centre is a cyclic selection machine partitioned into two zones:

* **Dark zone.**  Selected B cells (centroblasts) divide.  The number of
  divisions is set by how much T-cell help they received in the previous
  round, which is the mechanism that converts help into clonal expansion.
  AID mutates the rearranged V region during these divisions.

* **Light zone.**  Centrocytes compete to capture antigen held intact on
  follicular dendritic cells.  Capture scales with BCR affinity.  Captured
  antigen is processed and presented on MHC-II to a limited pool of Tfh
  cells, and the cells presenting most get the help.  Help is the scarce
  resource, so selection is *relative*: a clone's fate depends on its
  competitors, not on an absolute affinity threshold.

Cells that get help recycle to the dark zone or exit as plasma cells or
memory B cells.  Cells that do not, die.  Running this loop is what produces
affinity maturation.

References for the structure of the model:
    Victora GD, Nussenzweig MC. "Germinal centers." Annu Rev Immunol
    30:429-457 (2012).
    Meyer-Hermann M, et al. "A theory of germinal center B cell selection,
    division, and exit." Cell Rep 2:162-174 (2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .shm import HypermutationEngine, Mutation
from .vdj import Rearrangement, translate


class Zone(str, Enum):
    DARK = "dark"
    LIGHT = "light"


class Fate(str, Enum):
    RECYCLED = "recycled"
    PLASMA = "plasma"
    MEMORY = "memory"
    APOPTOSIS = "apoptosis"
    STOP_CODON = "stop_codon"


@dataclass
class BCell:
    """One germinal-centre B cell."""

    cell_id: int
    clone_id: int
    parent_id: Optional[int]
    rearrangement: Rearrangement
    seq_nt: str
    regions: list[str]
    isotype: str = "IgM"
    generation: int = 0
    birth_round: int = 0
    mutations: list[Mutation] = field(default_factory=list)
    log10_kd: float = 0.0
    help_received: float = 0.0
    zone: Zone = Zone.DARK
    #: Amino-acid indices of each CDR loop, computed once per founder clone
    #: and inherited by reference through the lineage.
    cdr_index: dict = field(default_factory=dict)

    def paratope(self) -> dict:
        aa = self.seq_aa
        return {
            loop: "".join(aa[i] for i in idx if i < len(aa))
            for loop, idx in self.cdr_index.items()
        }

    @property
    def seq_aa(self) -> str:
        return translate(self.seq_nt)

    @property
    def n_mutations(self) -> int:
        return len(self.mutations)

    @property
    def n_replacement(self) -> int:
        return sum(1 for m in self.mutations if m.replacement)

    @property
    def n_silent(self) -> int:
        return sum(1 for m in self.mutations if m.silent)

    @property
    def n_fr_replacement(self) -> int:
        return sum(1 for m in self.mutations if m.replacement and m.region.startswith("fwr"))

    @property
    def n_cdr_replacement(self) -> int:
        return sum(1 for m in self.mutations if m.replacement and m.region.startswith("cdr"))


@dataclass
class RoundSummary:
    """Population state at the end of one dark-zone/light-zone cycle."""

    round_index: int
    n_entering: int
    n_after_division: int
    n_selected: int
    n_apoptosis: int
    n_stop_codon: int
    n_plasma_out: int
    n_memory_out: int
    best_log10_kd: float
    median_log10_kd: float
    mean_log10_kd: float
    mean_mutations: float
    mean_cdr_replacement: float
    mean_fr_replacement: float
    n_clones: int
    shannon_diversity: float
    top_clone_fraction: float
    isotype_counts: dict


class GerminalCenter:
    """A single germinal centre running mutation-selection cycles."""

    def __init__(
        self,
        oracle,
        epitope_seq: str,
        shm_engine: Optional[HypermutationEngine] = None,
        class_switch=None,
        carrying_capacity: int = 2_500,
        max_light_zone_cells: int = 12_000,
        divisions_min: int = 1,
        divisions_max: int = 6,
        recycle_probability: float = 0.80,
        memory_fraction_of_exits: float = 0.45,
        memory_bias_early_rounds: int = 6,
        tfh_help_slots: Optional[int] = None,
        help_competition_sharpness: float = 1.6,
        fdc_antigen_molar: float = 1e-8,
        antigen_decay_per_round: float = 0.90,
        min_antigen_molar: float = 1e-12,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.oracle = oracle
        self.epitope_seq = epitope_seq
        self.shm = shm_engine or HypermutationEngine(rng=rng)
        self.csr = class_switch
        self.capacity = carrying_capacity
        self.max_light_zone_cells = max_light_zone_cells
        self.div_min = divisions_min
        self.div_max = divisions_max
        self.p_recycle = recycle_probability
        self.memory_fraction = memory_fraction_of_exits
        self.memory_bias_rounds = memory_bias_early_rounds
        self.help_slots = tfh_help_slots
        self.sharpness = help_competition_sharpness
        self.antigen_molar = fdc_antigen_molar
        self.antigen_decay = antigen_decay_per_round
        self.min_antigen_molar = min_antigen_molar
        self.rng = rng if rng is not None else np.random.default_rng()

        self.population: list[BCell] = []
        self.plasma_cells: list[BCell] = []
        self.memory_cells: list[BCell] = []
        self.history: list[RoundSummary] = []
        self._next_id = 0
        self.round_index = 0

    # -- setup --------------------------------------------------------------
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    @staticmethod
    def cdr_index_from_regions(regions: list[str]) -> dict:
        """Amino-acid positions of each CDR loop, from nucleotide labels."""
        out: dict[str, list[int]] = {"cdr1": [], "cdr2": [], "cdr3": []}
        for aa_pos in range(len(regions) // 3):
            lab = regions[aa_pos * 3]
            if lab in out:
                out[lab].append(aa_pos)
        return out

    def score(self, cell: BCell) -> float:
        res = self.oracle.score(cell.paratope(), self.epitope_seq,
                                framework_replacements=cell.n_fr_replacement)
        return res.log10_kd

    def seed(self, founders: list[BCell]) -> None:
        """Populate the germinal centre with founder clones."""
        for c in founders:
            if not c.cdr_index:
                c.cdr_index = self.cdr_index_from_regions(c.regions)
            c.log10_kd = self.score(c)
        self.population = list(founders)

    # -- one cycle ----------------------------------------------------------
    def _antigen_capture(self, log10_kd: float) -> float:
        """Amount of antigen a centrocyte strips off the FDC.

        Langmuir occupancy of the BCR at the current FDC antigen
        concentration: ``Ka*[Ag] / (1 + Ka*[Ag])``.  Because the half-
        occupancy point sits at ``KD == [Ag]``, the affinity that selection
        can still discriminate rises as FDC antigen is consumed — which is
        why maturation keeps making progress instead of stalling once the
        population passes the starting midpoint.
        """
        ka = 10.0 ** (-log10_kd)
        x = ka * self.antigen_molar
        return x / (1.0 + x)

    def _divisions_for(self, help_received: float) -> int:
        """Help-dependent division count in the dark zone."""
        span = self.div_max - self.div_min
        return int(self.div_min + round(span * float(np.clip(help_received, 0.0, 1.0))))

    def step(self, tfh_total_help: float = 1.0) -> RoundSummary:
        """Run one dark-zone/light-zone cycle."""
        self.round_index += 1
        # FDC-held antigen is consumed over the life of the response.
        if self.round_index > 1:
            self.antigen_molar = max(self.min_antigen_molar,
                                     self.antigen_molar * self.antigen_decay)
        n_entering = len(self.population)
        if n_entering == 0:
            raise RuntimeError("germinal centre is empty")

        # ---- dark zone: proliferate and mutate ----
        # The light zone can only process so many centrocytes per cycle, so
        # the brood is scaled down uniformly rather than letting the
        # population explode as 2**divisions.
        projected = sum(
            2 ** self._divisions_for(c.help_received if self.round_index > 1 else 0.5)
            for c in self.population
        )
        budget = self.max_light_zone_cells
        scale = min(1.0, budget / projected) if projected else 1.0

        offspring: list[BCell] = []
        n_stop = 0
        for cell in self.population:
            n_div = self._divisions_for(cell.help_received if self.round_index > 1 else 0.5)
            expected = (2 ** n_div) * scale
            n_children = int(self.rng.poisson(expected)) if expected < 1.0 else int(round(expected))
            for _ in range(n_children):
                new_nt, muts = self.shm.mutate(cell.seq_nt, cell.regions, divisions=n_div)
                child = BCell(
                    cell_id=self._new_id(),
                    clone_id=cell.clone_id,
                    parent_id=cell.cell_id,
                    rearrangement=cell.rearrangement,
                    seq_nt=new_nt,
                    regions=cell.regions,
                    isotype=cell.isotype,
                    generation=cell.generation + n_div,
                    birth_round=self.round_index,
                    mutations=cell.mutations + muts,
                    cdr_index=cell.cdr_index,
                )
                if "*" in child.seq_aa:
                    n_stop += 1
                    continue
                if self.csr is not None:
                    ev = self.csr.maybe_switch(child.isotype, self.round_index)
                    if ev is not None:
                        child.isotype = ev.to_isotype
                offspring.append(child)

        n_after_division = len(offspring)
        if not offspring:
            raise RuntimeError("all offspring carried stop codons")

        # ---- light zone: capture antigen, compete for Tfh help ----
        for c in offspring:
            c.log10_kd = self.score(c)
            c.zone = Zone.LIGHT

        capture = np.array([self._antigen_capture(c.log10_kd) for c in offspring])
        # Relative competition for a limited number of help events.
        w = capture ** self.sharpness
        if w.sum() <= 0:
            w = np.ones_like(w)
        p = w / w.sum()

        slots = self.help_slots if self.help_slots is not None else int(self.capacity * tfh_total_help)
        slots = max(1, min(slots, len(offspring)))
        winners_idx = self.rng.choice(len(offspring), size=slots, replace=False, p=p)
        winners = set(int(i) for i in winners_idx)

        selected: list[BCell] = []
        n_apoptosis = 0
        for i, c in enumerate(offspring):
            if i in winners:
                c.help_received = float(capture[i])
                selected.append(c)
            else:
                n_apoptosis += 1

        # ---- exits: recycle, plasma, memory ----
        recycled: list[BCell] = []
        plasma_out, memory_out = 0, 0
        early = self.round_index <= self.memory_bias_rounds
        mem_frac = self.memory_fraction * (1.6 if early else 0.6)
        for c in selected:
            if self.rng.random() < self.p_recycle:
                c.zone = Zone.DARK
                recycled.append(c)
            elif self.rng.random() < mem_frac:
                self.memory_cells.append(c)
                memory_out += 1
            else:
                self.plasma_cells.append(c)
                plasma_out += 1

        # ---- carrying capacity ----
        if len(recycled) > self.capacity:
            keep = self.rng.choice(len(recycled), size=self.capacity, replace=False)
            recycled = [recycled[int(i)] for i in keep]
        if not recycled:
            recycled = selected[: max(1, self.capacity // 10)]
        self.population = recycled

        summary = self._summarise(n_entering, n_after_division, len(selected),
                                  n_apoptosis, n_stop, plasma_out, memory_out)
        self.history.append(summary)
        return summary

    # -- reporting ----------------------------------------------------------
    def _summarise(self, n_entering, n_after_division, n_selected,
                   n_apoptosis, n_stop, plasma_out, memory_out) -> RoundSummary:
        pop = self.population
        kds = np.array([c.log10_kd for c in pop])
        muts = np.array([c.n_mutations for c in pop])
        clones = {}
        isotypes: dict[str, int] = {}
        for c in pop:
            clones[c.clone_id] = clones.get(c.clone_id, 0) + 1
            isotypes[c.isotype] = isotypes.get(c.isotype, 0) + 1
        counts = np.array(list(clones.values()), dtype=float)
        frac = counts / counts.sum()
        shannon = float(-(frac * np.log(frac)).sum())
        return RoundSummary(
            round_index=self.round_index,
            n_entering=n_entering,
            n_after_division=n_after_division,
            n_selected=n_selected,
            n_apoptosis=n_apoptosis,
            n_stop_codon=n_stop,
            n_plasma_out=plasma_out,
            n_memory_out=memory_out,
            best_log10_kd=float(kds.min()),
            median_log10_kd=float(np.median(kds)),
            mean_log10_kd=float(kds.mean()),
            mean_mutations=float(muts.mean()),
            mean_cdr_replacement=float(np.mean([c.n_cdr_replacement for c in pop])),
            mean_fr_replacement=float(np.mean([c.n_fr_replacement for c in pop])),
            n_clones=len(clones),
            shannon_diversity=shannon,
            top_clone_fraction=float(frac.max()),
            isotype_counts=isotypes,
        )

    def run(self, n_rounds: int, tfh_total_help: float = 1.0, verbose: bool = False):
        for _ in range(n_rounds):
            s = self.step(tfh_total_help=tfh_total_help)
            if verbose:
                print(f"round {s.round_index:>2}  n={len(self.population):>5}  "
                      f"best={s.best_log10_kd:6.2f}  median={s.median_log10_kd:6.2f}  "
                      f"mut={s.mean_mutations:5.1f}  clones={s.n_clones:>4}")
        return self.history


__all__ = ["GerminalCenter", "BCell", "RoundSummary", "Zone", "Fate"]
