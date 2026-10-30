"""Developability gates — turning affinity into a go/no-go decision.

Most antibody programmes die after binding is solved, on manufacturability,
stability and safety.  A twin that only reports affinity models the easy
half of the problem, so this module scores the eight-dimension panel used
industrially and returns a tier rather than a number:

    A — advance,  B — optimise,  C — reject or redesign.

.. warning::

   These are **heuristic gates computed from sequence**, not predictions.
   No published model predicts all eight properties at clinical-grade
   accuracy; the panel exists so the twin's output funnel has the same
   *shape* as a real campaign's, with attrition where real attrition
   happens.

Two distinct criterion sets are applied, and they should not be conflated.

**Eight-dimension property panel** — the predictive dimensions, each with
a commonly quoted industrial threshold:

   ===========================  ==================================
   dimension                    threshold
   ===========================  ==================================
   binding                      KD < 1 nM
   specificity                  < 10% cross-reactivity
   aggregation                  < 5% at 100 mg/mL
   viscosity                    < 20 cP at 150 mg/mL
   immunogenicity               < 5 predicted MHC-II epitopes
   thermal stability            Tm > 65 °C
   solubility                   > 50 mg/mL
   manufacturability            > 1 g/L CHO titre
   ===========================  ==================================

**Release-style criteria** — a separate, measurement-oriented set with its
own, lower stability bar:

   ===========================  ==================================
   nanoDSF lowest-domain Tm     >= 58 °C
   monomer content              >= 90%
   isoelectric point            6.0 - 9.0
   ===========================  ==================================

The two stability numbers differ on purpose: 65 °C is the property-panel
target, 58 °C the lowest-domain floor below which a candidate is rejected
outright.  Falling between them is a warning, not a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class Tier(str, Enum):
    ADVANCE = "A"
    OPTIMISE = "B"
    REJECT = "C"


#: Sequence liability motifs scanned in the variable domain.
LIABILITY_MOTIFS = {
    "n_glycosylation": r"N[^P][ST]",
    "deamidation_fast": r"N[GS]",
    "isomerisation": r"D[GSTD]",
    "fragmentation": r"DP",
    "oxidation_met": r"M",
    "free_cysteine": r"C",
    "tryptophan_oxidation": r"W",
}

#: The eight predictive dimensions and their industrial thresholds.
PROPERTY_PANEL = {
    "kd_molar": 1e-9,
    "cross_reactivity": 0.10,
    "aggregation_pct": 5.0,
    "viscosity_cp": 20.0,
    "mhcii_epitopes": 5,
    "tm_celsius": 65.0,
    "solubility_mg_ml": 50.0,
    "titre_g_l": 1.0,
}

#: Release-style measurement criteria, applied alongside the property
#: panel.  ``tm_lowest_domain`` is the nanoDSF floor, deliberately lower
#: than the property panel's 65 C target.
RELEASE_CRITERIA = {
    "tm_lowest_domain": 58.0,
    "monomer_pct": 90.0,
    "pi_low": 6.0,
    "pi_high": 9.0,
}

#: Convenience union of both criterion sets.
THRESHOLDS = {**PROPERTY_PANEL, **RELEASE_CRITERIA}


@dataclass
class DevelopabilityReport:
    """Per-dimension outcome for one candidate."""

    tier: Tier
    kd_molar: float
    cross_reactivity: float
    aggregation_pct: float
    viscosity_cp: float
    mhcii_epitopes: int
    tm_celsius: float
    solubility_mg_ml: float
    titre_g_l: float
    monomer_pct: float
    isoelectric_point: float
    gravy: float
    instability_index: float
    liabilities: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def passed(self) -> bool:
        return self.tier is Tier.ADVANCE


class DevelopabilityPanel:
    """Scores a variable-domain sequence against the eight-dimension panel."""

    def __init__(self, apc_module=None, n_fail_for_reject: int = 3) -> None:
        self.apc = apc_module
        self.n_fail_for_reject = n_fail_for_reject
        # Hydrophobic-patch reference, set by calibrate().  Until then the
        # defaults place a typical human germline CDR set at patch_z = 0.
        self._patch_ref = 0.55
        self._patch_spread = 0.08

    def calibrate(self, germline_paratopes, patch_quantile: float = 0.5) -> dict:
        """Anchor the patch scale on unmutated human germline CDR loops.

        The raw hydrophobic-patch score is an internal quantity with no
        absolute meaning: every antibody scores around 0.55 on it because
        CDR loops are hydrophobic by construction.  Thresholds therefore
        have to be expressed as deviation from a baseline, and the only
        defensible baseline is the human germline repertoire — sequences
        that evolution has already filtered for being expressible and
        soluble.  After calibration, the median germline antibody sits at
        the centre of every property scale and only genuine outliers fail.
        """
        import numpy as _np

        vals = _np.array([
            self._hydrophobic_patch("".join(p.get(k, "") for k in ("cdr1", "cdr2", "cdr3")))
            for p in germline_paratopes
        ])
        self._patch_ref = float(_np.quantile(vals, patch_quantile))
        spread = float(_np.quantile(vals, 0.84) - _np.quantile(vals, 0.16)) / 2.0
        self._patch_spread = spread if spread > 1e-6 else 0.08
        return {
            "n_sampled": int(vals.size),
            "patch_reference": self._patch_ref,
            "patch_spread": self._patch_spread,
        }

    # -- individual heuristics ---------------------------------------------
    @staticmethod
    def _hydrophobic_patch(cdr_seq: str) -> float:
        """Largest run-weighted hydrophobic content across the CDR loops."""
        hyd = set("FWYLIVMA")
        best, run = 0, 0
        for aa in cdr_seq:
            run = run + 1 if aa in hyd else 0
            best = max(best, run)
        frac = sum(1 for a in cdr_seq if a in hyd) / max(1, len(cdr_seq))
        return 0.6 * frac + 0.4 * min(1.0, best / 6.0)

    def _count_mhcii(self, seq: str) -> int:
        """Number of strong-ish MHC-II binding cores, by the APC heuristic."""
        if self.apc is None:
            return 0
        scores = []
        for i in range(0, max(1, len(seq) - 15 + 1)):
            pep = seq[i:i + 15]
            best = -np.inf
            for j in range(0, max(1, len(pep) - 9 + 1)):
                best = max(best, self.apc._core_score(pep[j:j + 9]))
            scores.append(best)
        if not scores:
            return 0
        arr = np.asarray(scores)
        # "Strong" = top decile of the intrinsic scale, counted in
        # non-overlapping windows so one good core is not counted 15 times.
        strong = arr > 10.0
        count, i = 0, 0
        while i < len(strong):
            if strong[i]:
                count += 1
                i += 9
            else:
                i += 1
        return int(count)

    # -- main entry point ---------------------------------------------------
    def evaluate(self, seq_aa: str, cdr_seqs: dict, kd_molar: float,
                 self_reactivity_log10_kd: Optional[float] = None) -> DevelopabilityReport:
        pa = ProteinAnalysis(seq_aa.replace("*", ""))
        pi = float(pa.isoelectric_point())
        gravy = float(pa.gravy())
        instability = float(pa.instability_index())
        charge = float(pa.charge_at_pH(7.4))

        cdr_concat = "".join(cdr_seqs.get(k, "") for k in ("cdr1", "cdr2", "cdr3"))
        patch_raw = self._hydrophobic_patch(cdr_concat)
        # Deviation from the germline baseline, in units of its own spread.
        patch_z = (patch_raw - self._patch_ref) / self._patch_spread
        # Only excess hydrophobicity is penalised; a more polar than
        # average paratope is not a manufacturability problem.
        patch = max(0.0, patch_z)

        liabilities = {
            name: len(re.findall(pat, seq_aa))
            for name, pat in LIABILITY_MOTIFS.items()
        }
        # Only unpaired cysteines beyond the two canonical ones are a liability.
        liabilities["free_cysteine"] = max(0, liabilities["free_cysteine"] - 2)

        # Heuristic property estimates.  `patch` here is the calibrated
        # deviation above the germline baseline, so patch == 0 means "a
        # median human germline paratope" (whose raw score is ~0.5, not 0).
        # Each estimate is anchored to sit comfortably inside the
        # industrial threshold at patch == 0, and degrades with excess
        # hydrophobicity, charge imbalance and predicted instability.
        tm = 71.0 - 3.2 * patch - 0.22 * max(0.0, instability - 38.0)
        aggregation = 1.8 + 1.5 * patch + 0.05 * max(0.0, abs(charge) - 8.0)
        viscosity = 9.0 + 3.6 * patch + 0.9 * max(0.0, abs(charge) - 6.0)
        solubility = 95.0 - 13.0 * patch - 3.0 * max(0.0, abs(charge) - 10.0)
        titre = max(0.05, 2.6 - 0.40 * patch - 0.02 * max(0.0, instability - 42.0))
        monomer = 99.0 - 1.6 * aggregation
        mhcii = self._count_mhcii(seq_aa)
        cross = float(np.clip(0.03 + 0.022 * patch, 0.0, 1.0))
        if self_reactivity_log10_kd is not None:
            cross = float(np.clip(cross + max(0.0, (-self_reactivity_log10_kd - 6.0) * 0.08), 0.0, 1.0))

        failures, warnings = [], []
        T = THRESHOLDS

        def gate(ok: bool, label: str, hard: bool = True) -> None:
            if not ok:
                (failures if hard else warnings).append(label)

        gate(kd_molar < T["kd_molar"], f"binding KD {kd_molar:.2e} M >= 1 nM")
        gate(cross < T["cross_reactivity"], f"cross-reactivity {cross:.0%}")
        gate(aggregation < T["aggregation_pct"], f"aggregation {aggregation:.1f}%")
        gate(viscosity < T["viscosity_cp"], f"viscosity {viscosity:.0f} cP")
        gate(mhcii < T["mhcii_epitopes"], f"{mhcii} MHC-II epitopes")
        gate(tm > T["tm_celsius"], f"Tm {tm:.0f} C", hard=tm <= T["tm_lowest_domain"])
        gate(solubility > T["solubility_mg_ml"], f"solubility {solubility:.0f} mg/mL")
        gate(titre > T["titre_g_l"], f"titre {titre:.2f} g/L")
        gate(monomer >= T["monomer_pct"], f"monomer {monomer:.1f}%")
        gate(T["pi_low"] <= pi <= T["pi_high"], f"pI {pi:.1f} outside 6.0-9.0", hard=False)
        if liabilities["n_glycosylation"]:
            warnings.append(f"{liabilities['n_glycosylation']} N-glycosylation site(s)")
        if liabilities["free_cysteine"]:
            failures.append(f"{liabilities['free_cysteine']} unpaired cysteine(s)")

        if not failures and not warnings:
            tier = Tier.ADVANCE
        elif len(failures) >= self.n_fail_for_reject:
            tier = Tier.REJECT
        elif failures:
            tier = Tier.OPTIMISE
        else:
            tier = Tier.ADVANCE if len(warnings) <= 1 else Tier.OPTIMISE

        return DevelopabilityReport(
            tier=tier, kd_molar=kd_molar, cross_reactivity=cross,
            aggregation_pct=aggregation, viscosity_cp=viscosity,
            mhcii_epitopes=mhcii, tm_celsius=tm, solubility_mg_ml=solubility,
            titre_g_l=titre, monomer_pct=monomer, isoelectric_point=pi,
            gravy=gravy, instability_index=instability,
            liabilities=liabilities, failures=failures, warnings=warnings,
        )


__all__ = ["DevelopabilityPanel", "DevelopabilityReport", "Tier",
           "THRESHOLDS", "PROPERTY_PANEL", "RELEASE_CRITERIA", "LIABILITY_MOTIFS"]
