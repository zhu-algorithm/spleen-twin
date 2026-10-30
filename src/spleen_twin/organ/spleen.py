"""The virtual spleen.

Anatomy represented
-------------------
================  ======================================================
compartment       what it does here
================  ======================================================
red pulp          filters blood, delivers antigen, houses long-lived
                  plasma cells
marginal zone     captures blood-borne antigen, holds MZ B cells that
                  give the fast extrafollicular response
PALS (T zone)     dendritic cells prime CD4+ T cells; Tfh differentiate
follicle          naive B cells; the germinal centre forms here, with a
                  dark zone (proliferation + SHM) and a light zone
                  (FDC antigen capture + Tfh selection)
================  ======================================================

The organ is the object that owns the chain antigen -> presentation ->
T-cell help -> B-cell activation -> V(D)J repertoire -> germinal centre ->
antibody.  Each step is a module that can be swapped independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..modules.activation import ActivationFate, ActivationModule
from ..modules.affinity import AffinityOracle, SurrogateAffinityOracle, paratope_from
from ..modules.antigen import Antigen
from ..modules.apc import AntigenPresentingCompartment
from ..modules.csr import ISOTYPE_HALF_LIFE_DAYS, ClassSwitchModule
from ..modules.germinal_center import BCell, GerminalCenter
from ..modules.regions import ImgtAnnotator
from ..modules.shm import HypermutationEngine
from ..modules.tcell import TCellCompartment
from ..modules.tolerance import ToleranceCheckpoint, ToleranceOutcome, pre_bcr_checkpoint
from ..modules.vdj import ChainRecombinator


@dataclass
class RepertoireStats:
    """What V(D)J and tolerance produced before any antigen arrived."""

    n_attempted: int
    n_productive: int
    n_annotated: int
    n_deleted: int
    n_edited: int
    n_anergic: int
    n_naive: int

    @property
    def productive_fraction(self) -> float:
        return self.n_productive / max(1, self.n_attempted)


@dataclass
class CampaignResult:
    """Everything one immunisation produced."""

    repertoire: RepertoireStats
    n_founders: int
    gc_history: list
    plasma_cells: list
    memory_cells: list
    serum_titre: list = field(default_factory=list)
    tfh: object = None
    presentation: object = None


class Spleen:
    """A digital-twin spleen driving one antibody campaign."""

    def __init__(
        self,
        oracle: Optional[AffinityOracle] = None,
        seed: int = 20261030,
        heavy_locus: str = "IGH",
        n_germinal_centers: int = 1,
        gc_kwargs: Optional[dict] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.seed_value = seed

        self.recombinator = ChainRecombinator(heavy_locus, rng=self.rng)
        self.annotator = ImgtAnnotator()
        self.shm = HypermutationEngine(rng=self.rng)
        self.apc = AntigenPresentingCompartment(rng=self.rng)
        self.tcells = TCellCompartment(rng=self.rng)
        self.activation = ActivationModule(rng=self.rng)
        self.csr = ClassSwitchModule(rng=self.rng)
        self.oracle = oracle or SurrogateAffinityOracle()

        self.n_gcs = n_germinal_centers
        self.gc_kwargs = gc_kwargs or {}
        self.germinal_centers: list[GerminalCenter] = []

        self.naive_pool: list[BCell] = []
        self.repertoire_stats: Optional[RepertoireStats] = None
        self._calibrated = False

    # -- 1. repertoire ------------------------------------------------------
    def generate_naive_repertoire(
        self,
        n_cells: int = 5_000,
        tolerance: Optional[ToleranceCheckpoint] = None,
    ) -> RepertoireStats:
        """Run V(D)J recombination and the tolerance checkpoints.

        Only cells that pass the pre-BCR checkpoint (productive heavy chain)
        and survive self-screening enter the naive pool.
        """
        attempted = productive = annotated = 0
        deleted = edited = anergic = 0
        pool: list[BCell] = []

        while len(pool) < n_cells and attempted < n_cells * 60:
            attempted += 1
            r = self.recombinator.rearrange()
            if not pre_bcr_checkpoint(r):
                continue
            productive += 1
            rm = self.annotator.annotate(r)
            if not rm.annotated:
                continue
            annotated += 1

            cell = BCell(
                cell_id=len(pool), clone_id=len(pool), parent_id=None,
                rearrangement=r, seq_nt=r.seq_nt, regions=rm.labels,
                cdr_index=GerminalCenter.cdr_index_from_regions(rm.labels),
            )
            if tolerance is not None:
                res = tolerance.screen(cell.paratope())
                if res.outcome is ToleranceOutcome.DELETED:
                    deleted += 1
                    continue
                if res.outcome is ToleranceOutcome.EDITED:
                    edited += 1
                    continue
                if res.outcome is ToleranceOutcome.ANERGIC:
                    anergic += 1
                    continue
            pool.append(cell)

        self.naive_pool = pool
        self.repertoire_stats = RepertoireStats(
            n_attempted=attempted, n_productive=productive, n_annotated=annotated,
            n_deleted=deleted, n_edited=edited, n_anergic=anergic, n_naive=len(pool),
        )
        return self.repertoire_stats

    def calibrate_oracle(self, epitope_seq: str, n: int = 2_000) -> None:
        """Anchor the affinity scale on this spleen's own naive repertoire."""
        if not self.naive_pool:
            raise RuntimeError("generate the naive repertoire first")
        sample = self.naive_pool[:n]
        raws = [self.oracle.raw_score(c.paratope(), epitope_seq)[0] for c in sample]
        self.oracle.calibrate(raws)
        self._calibrated = True

    # -- 2..4. presentation, help, activation -------------------------------
    def immunise(
        self,
        antigen: Antigen,
        epitope_name: Optional[str] = None,
        n_rounds: int = 30,
        max_founders: int = 150,
    ) -> CampaignResult:
        """Run one complete immunisation through the whole organ."""
        if not self.naive_pool:
            self.generate_naive_repertoire()

        epitope = (
            next(e for e in antigen.epitopes if e.name == epitope_name)
            if epitope_name else antigen.immunodominant()
        )
        if not self._calibrated:
            self.calibrate_oracle(epitope.sequence)

        # red pulp / marginal zone deliver antigen; APCs process and load FDCs
        presentation = self.apc.present(antigen)
        # PALS: prime CD4+ T cells and make Tfh
        tfh = self.tcells.prime(presentation)

        # T-B border: which naive B cells are activated and where they go
        founders: list[BCell] = []
        for cell in self.naive_pool:
            kd = self.oracle.score(cell.paratope(), epitope.sequence).log10_kd
            cell.log10_kd = kd
            act = self.activation.activate(kd, tfh=tfh, help_available=tfh.quality)
            if act.fate is ActivationFate.GERMINAL_CENTER:
                founders.append(cell)
            if len(founders) >= max_founders:
                break

        if not founders:
            raise RuntimeError("no naive B cell cleared the activation threshold")

        # follicle: seed germinal centres and cycle them
        self.germinal_centers = []
        per_gc = max(1, len(founders) // self.n_gcs)
        plasma, memory, histories = [], [], []
        for g in range(self.n_gcs):
            subset = founders[g * per_gc:(g + 1) * per_gc] or founders[:per_gc]
            gc = GerminalCenter(
                self.oracle, epitope.sequence,
                shm_engine=self.shm, class_switch=self.csr,
                fdc_antigen_molar=presentation.fdc_antigen_nm * 1e-9,
                rng=self.rng, **self.gc_kwargs,
            )
            gc.seed([
                BCell(cell_id=c.cell_id, clone_id=c.clone_id, parent_id=None,
                      rearrangement=c.rearrangement, seq_nt=c.seq_nt,
                      regions=c.regions, cdr_index=c.cdr_index)
                for c in subset
            ])
            gc.run(n_rounds, tfh_total_help=min(1.0, tfh.quality + 0.3))
            self.germinal_centers.append(gc)
            plasma.extend(gc.plasma_cells)
            memory.extend(gc.memory_cells)
            histories.append(gc.history)

        titre = self.serum_titre(plasma, n_rounds)
        return CampaignResult(
            repertoire=self.repertoire_stats,
            n_founders=len(founders),
            gc_history=histories[0] if self.n_gcs == 1 else histories,
            plasma_cells=plasma,
            memory_cells=memory,
            serum_titre=titre,
            tfh=tfh,
            presentation=presentation,
        )

    # -- 5. antibody output -------------------------------------------------
    @staticmethod
    def serum_titre(plasma_cells: list[BCell], n_rounds: int,
                    hours_per_round: float = 8.0,
                    secretion_per_cell_per_day: float = 1.0) -> list[dict]:
        """Accumulate secreted antibody with isotype-specific clearance.

        One germinal-centre cycle is taken as ``hours_per_round``; plasma
        cells secrete from the round they exit until the end of the run and
        their product decays with the serum half-life of its isotype.
        """
        days_per_round = hours_per_round / 24.0
        series = []
        total = 0.0
        weighted_affinity_num = 0.0
        for r in range(1, n_rounds + 1):
            # decay everything already in serum
            new_total, new_aff = 0.0, 0.0
            for c in plasma_cells:
                if c.birth_round > r:
                    continue
                age_days = (r - c.birth_round) * days_per_round
                hl = ISOTYPE_HALF_LIFE_DAYS.get(c.isotype, 14.0)
                amount = secretion_per_cell_per_day * days_per_round * (0.5 ** (age_days / hl))
                new_total += amount
                new_aff += amount * (-c.log10_kd)
            total, weighted_affinity_num = new_total, new_aff
            series.append({
                "round": r,
                "titre_au": total,
                "mean_pkd": (weighted_affinity_num / total) if total > 0 else 0.0,
            })
        return series

    # -- 6. harvest ---------------------------------------------------------
    def harvest_single_b_cells(
        self,
        result: CampaignResult,
        n_sorted: int = 2_000,
        antigen_positive_rate: float = 0.10,
        export_rate: float = 0.85,
        pairing_success: float = 0.90,
    ) -> list[BCell]:
        """Simulate the single-B-cell screening funnel.

        Default yields follow commonly reported platform figures: roughly
        10% of sorted cells are antigen-positive, >85% of those export, and
        >90% give a usable heavy/light pair.
        """
        pool = result.plasma_cells + result.memory_cells
        if not pool:
            return []
        n_sorted = min(n_sorted, len(pool))
        idx = self.rng.choice(len(pool), size=n_sorted, replace=False)
        sorted_cells = [pool[int(i)] for i in idx]

        n_pos = int(self.rng.binomial(n_sorted, antigen_positive_rate))
        # Antigen-positive sorting enriches for the better binders.
        sorted_cells.sort(key=lambda c: c.log10_kd)
        positives = sorted_cells[:n_pos]

        exported = [c for c in positives if self.rng.random() < export_rate]
        paired = [c for c in exported if self.rng.random() < pairing_success]
        return paired


__all__ = ["Spleen", "CampaignResult", "RepertoireStats"]
