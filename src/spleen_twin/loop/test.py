"""Test — measure the proposals.

In this repository the measurement is the twin's own oracle, which closes
the loop entirely in silico.  In a deployment this module is the interface
to the wet lab: same signature, with the oracle replaced by an assay
result read back from a LIMS.  Keeping that boundary explicit is what
makes the loop swappable rather than a simulation quietly treating itself
as ground truth.
"""

from __future__ import annotations

from typing import Optional

from ..modules.developability import DevelopabilityPanel
from .curate import AntibodyRecord
from .design import Proposal


def measure_proposals(
    proposals: list,
    oracle,
    epitope_seq: str,
    metadata: dict,
    round_index: int,
    panel: Optional[DevelopabilityPanel] = None,
    binder_threshold_log10_kd: float = -7.0,
) -> list:
    """Score proposals and return them in curated-record shape."""
    out = []
    for i, p in enumerate(proposals):
        paratope = {"cdr1": p.cdr1_aa, "cdr2": p.cdr2_aa, "cdr3": p.cdr3_aa}
        res = oracle.score(paratope, epitope_seq)
        rec = AntibodyRecord(
            sequence_id=f"r{round_index}-{p.proposer}-{i:05d}",
            sequence="", sequence_aa=p.sequence_aa,
            v_call=p.v_call, d_call=None, j_call="",
            junction_aa=p.cdr3_aa, productive=True,
            cdr1_aa=p.cdr1_aa, cdr2_aa=p.cdr2_aa, cdr3_aa=p.cdr3_aa,
            n_v_mutations=p.n_v_mutations,
            kd_molar=res.kd_molar, log10_kd=res.log10_kd,
            binder=res.log10_kd <= binder_threshold_log10_kd,
            metadata={**metadata, "proposer": p.proposer,
                      "predicted_log10_kd": p.predicted_log10_kd,
                      "loop_round": round_index},
            label_source="simulated",
        )
        if panel is not None:
            rep = panel.evaluate(p.sequence_aa, paratope, res.kd_molar)
            rec.expression_g_l = rep.titre_g_l
            rec.tm_celsius = rep.tm_celsius
            rec.developability_tier = rep.tier.value
        out.append(rec)
    return out


__all__ = ["measure_proposals"]
