"""Multimodal AI contracts for regulatory-aware antibody maturation.

The module deliberately separates four predictions that are often conflated:

* GeneFormer-X represents the B-cell state and its regulatory programme.
* A V(D)J sequence model represents paired IGH and IGK/IGL sequences.
* A diffusion model proposes conditional heavy- and light-chain mutations.
* A structure backend predicts an antibody-antigen complex and confidence.

An AlphaFold-like structure is evidence about geometry, not a dissociation
constant.  Affinity must come from a separately calibrated energy/affinity
backend and ultimately from experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RegulatoryState:
    cell_embedding: tuple[float, ...]
    pathway_scores: Mapping[str, float] = field(default_factory=dict)
    uncertainty: float | None = None


@dataclass(frozen=True)
class PairedSequence:
    heavy_aa: str
    light_aa: str
    heavy_v: str = ""
    heavy_d: str = ""
    heavy_j: str = ""
    light_v: str = ""
    light_j: str = ""


@dataclass(frozen=True)
class SequenceProposal:
    sequence: PairedSequence
    log_probability: float
    parent_id: str = ""
    changed_heavy_positions: tuple[int, ...] = ()
    changed_light_positions: tuple[int, ...] = ()


@dataclass(frozen=True)
class StructureAssessment:
    complex_confidence: float
    interface_confidence: float
    clash_score: float
    interface_area: float | None = None
    provenance: str = ""


@dataclass(frozen=True)
class RankedProposal:
    proposal: SequenceProposal
    structure: StructureAssessment
    predicted_log10_kd: float | None
    affinity_uncertainty: float | None
    selection_score: float


class GeneFormerX(Protocol):
    """Gene-centric, regulatory-aware encoder for a single B-cell state."""

    def encode(self, expression: Mapping[str, float],
               regulatory_edges: Sequence[tuple[str, str]]) -> RegulatoryState: ...


class VDJSequenceModel(Protocol):
    """Paired-chain model retaining V/D/J calls, junctions and CDR context."""

    def propose(self, parent: PairedSequence, state: RegulatoryState,
                antigen_sequence: str, n: int) -> Sequence[SequenceProposal]: ...


class AntibodyDiffusionModel(Protocol):
    """Refine candidates while conditioning jointly on both antibody chains."""

    def refine(self, proposals: Sequence[SequenceProposal], antigen_sequence: str,
               state: RegulatoryState) -> Sequence[SequenceProposal]: ...


class ComplexStructureModel(Protocol):
    """AlphaFold-like antibody-antigen complex prediction backend."""

    def assess(self, antibody: PairedSequence,
               antigen_sequence: str) -> StructureAssessment: ...


class CalibratedAffinityModel(Protocol):
    """Affinity head calibrated against measured KD/kon/koff, not pLDDT."""

    def predict(self, antibody: PairedSequence, antigen_sequence: str,
                structure: StructureAssessment) -> tuple[float, float | None]: ...


class RegulatoryAwareMaturationModel:
    """Compose cell state, paired VDJ, diffusion, structure and affinity models."""

    def __init__(self, geneformer: GeneFormerX, vdj_model: VDJSequenceModel,
                 diffusion: AntibodyDiffusionModel,
                 structure: ComplexStructureModel,
                 affinity: CalibratedAffinityModel) -> None:
        self.geneformer = geneformer
        self.vdj_model = vdj_model
        self.diffusion = diffusion
        self.structure = structure
        self.affinity = affinity

    def propose_and_rank(
        self,
        parent: PairedSequence,
        antigen_sequence: str,
        expression: Mapping[str, float],
        regulatory_edges: Sequence[tuple[str, str]],
        n: int = 32,
    ) -> list[RankedProposal]:
        state = self.geneformer.encode(expression, regulatory_edges)
        proposals = self.vdj_model.propose(parent, state, antigen_sequence, n)
        proposals = self.diffusion.refine(proposals, antigen_sequence, state)
        ranked: list[RankedProposal] = []
        for proposal in proposals:
            structure = self.structure.assess(proposal.sequence, antigen_sequence)
            logkd, uncertainty = self.affinity.predict(
                proposal.sequence, antigen_sequence, structure
            )
            # Lower log10(KD) is better. Structure confidence is a gate/regularizer,
            # never treated as a substitute for affinity.
            score = (-logkd + 0.5 * structure.interface_confidence
                     + 0.25 * structure.complex_confidence
                     - structure.clash_score
                     + 0.05 * proposal.log_probability)
            ranked.append(RankedProposal(proposal, structure, logkd, uncertainty, score))
        return sorted(ranked, key=lambda item: item.selection_score, reverse=True)


__all__ = [
    "RegulatoryState", "PairedSequence", "SequenceProposal",
    "StructureAssessment", "RankedProposal", "GeneFormerX",
    "VDJSequenceModel", "AntibodyDiffusionModel", "ComplexStructureModel",
    "CalibratedAffinityModel", "RegulatoryAwareMaturationModel",
]

