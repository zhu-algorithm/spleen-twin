"""Versioned, evidence-aware antibody candidate cards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class CandidateCard:
    candidate_id: str
    heavy_sequence_aa: str
    light_sequence_aa: str
    heavy_v_call: str
    heavy_d_call: Optional[str]
    heavy_j_call: str
    light_v_call: str
    light_j_call: str
    isotype: str
    clone_id: int
    parent_id: Optional[int]
    n_mutations: int
    affinity_log10_kd: float
    affinity_source: str
    affinity_uncertainty_log10: Optional[float]
    developability_tier: str
    developability_failures: list[str] = field(default_factory=list)
    developability_warnings: list[str] = field(default_factory=list)
    editing_history: list[dict] = field(default_factory=list)
    schema_version: str = "candidate-card-1"

    def to_dict(self) -> dict:
        return asdict(self)


def card_from_cell(cell, oracle, epitope_seq: str, panel) -> CandidateCard:
    if cell.light_rearrangement is None:
        raise ValueError("candidate cards require a paired receptor")
    affinity = oracle.score(cell.paratope(), epitope_seq,
                            framework_replacements=cell.n_fr_replacement)
    paired_aa = cell.seq_aa + cell.light_rearrangement.seq_aa
    dev = panel.evaluate(paired_aa, cell.paratope(), affinity.kd_molar)
    return CandidateCard(
        candidate_id=f"clone-{cell.clone_id}-cell-{cell.cell_id}",
        heavy_sequence_aa=cell.seq_aa,
        light_sequence_aa=cell.light_rearrangement.seq_aa,
        heavy_v_call=cell.rearrangement.v_gene,
        heavy_d_call=cell.rearrangement.d_gene,
        heavy_j_call=cell.rearrangement.j_gene,
        light_v_call=cell.light_rearrangement.v_gene,
        light_j_call=cell.light_rearrangement.j_gene,
        isotype=cell.isotype, clone_id=cell.clone_id, parent_id=cell.parent_id,
        n_mutations=cell.n_mutations, affinity_log10_kd=affinity.log10_kd,
        affinity_source=affinity.source_type,
        affinity_uncertainty_log10=affinity.uncertainty_log10,
        developability_tier=dev.tier.value,
        developability_failures=list(dev.failures),
        developability_warnings=list(dev.warnings),
        editing_history=[asdict(e) for e in cell.editing_history],
    )


def lineage_edges(cells) -> list[dict]:
    """Lossless edge list suitable for JSON/graph tools."""
    return [{"cell_id": c.cell_id, "parent_id": c.parent_id, "clone_id": c.clone_id,
             "generation": c.generation, "birth_round": c.birth_round,
             "mutations": [asdict(m) for m in c.mutations], "isotype": c.isotype}
            for c in cells]


__all__ = ["CandidateCard", "card_from_cell", "lineage_edges"]
