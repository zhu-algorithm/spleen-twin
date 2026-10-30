"""Generate — run the organ and emit labelled observations.

This is the loop's wet-lab surrogate.  It runs a campaign through the
virtual spleen and turns the harvested clones into curated records,
deliberately including the ones that failed: non-binders, poor expressers
and developability rejects.  A loop trained only on winners cannot learn
where the cliff is.
"""

from __future__ import annotations

from typing import Optional

from ..modules.developability import DevelopabilityPanel
from .curate import AntibodyRecord


def campaign_metadata(campaign_id: str, spleen, antigen, epitope_name: str,
                      schedule: str = "prime d0, boost d14, boost d28") -> dict:
    """Provenance block attached to every record from one campaign."""
    return {
        "campaign_id": campaign_id,
        "subject_id": f"virtual-spleen-seed-{spleen.seed_value}",
        "immunisation_schedule": schedule,
        "antigen_id": antigen.name,
        "antigen_lot": f"insilico-{antigen.name}-{len(antigen.sequence)}aa",
        "antigen_conformation": "monomeric, native sequence",
        "assay": "simulated",
        "assay_conditions": f"surrogate affinity oracle, 310 K, epitope {epitope_name}",
        "fitting_model": ("surrogate physicochemical complementarity, "
                          "two-anchor calibration to naive repertoire"),
        "raw_record_uri": f"spleen-twin://{campaign_id}",
    }


def records_from_campaign(
    campaign_id: str,
    spleen,
    antigen,
    result,
    epitope_name: str,
    panel: Optional[DevelopabilityPanel] = None,
    binder_threshold_log10_kd: float = -7.0,
    n_sorted: int = 2_000,
    include_unsorted_negatives: int = 500,
) -> list[AntibodyRecord]:
    """Harvest a campaign into learnable records, negatives included."""
    panel = panel or DevelopabilityPanel(apc_module=spleen.apc)
    meta = campaign_metadata(campaign_id, spleen, antigen, epitope_name)

    harvested = spleen.harvest_single_b_cells(result, n_sorted=n_sorted)
    hit_ids = {id(c) for c in harvested}
    pool = result.plasma_cells + result.memory_cells
    leftovers = [c for c in pool if id(c) not in hit_ids][:include_unsorted_negatives]

    records: list[AntibodyRecord] = []
    for tag, cells in (("hit", harvested), ("neg", leftovers)):
        for i, c in enumerate(cells):
            cdrs = c.paratope()
            kd = 10.0 ** c.log10_kd
            rep = panel.evaluate(c.seq_aa, cdrs, kd)
            records.append(AntibodyRecord(
                sequence_id=f"{campaign_id}-{tag}-{i:05d}",
                sequence=c.seq_nt,
                sequence_aa=c.seq_aa,
                v_call=c.rearrangement.v_gene,
                d_call=c.rearrangement.d_gene,
                j_call=c.rearrangement.j_gene,
                junction_aa=c.rearrangement.cdr3_aa,
                productive=True,
                cdr1_aa=cdrs.get("cdr1", ""),
                cdr2_aa=cdrs.get("cdr2", ""),
                cdr3_aa=cdrs.get("cdr3", ""),
                n_v_mutations=c.n_mutations,
                kd_molar=kd,
                log10_kd=c.log10_kd,
                binder=c.log10_kd <= binder_threshold_log10_kd,
                expression_g_l=rep.titre_g_l,
                tm_celsius=rep.tm_celsius,
                developability_tier=rep.tier.value,
                metadata=dict(meta),
                label_source="simulated",
            ))
    return records


__all__ = ["records_from_campaign", "campaign_metadata"]
