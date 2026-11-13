"""Virtual bone-marrow B-cell development for multimodal model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

from ..modules.ai_maturation import PairedSequence, RegulatoryState, StructureAssessment
from ..modules.alphagenome_vdj import AlphaGenome2VDJPrior


class BCellStage(str, Enum):
    HSC = "HSC"
    CLP = "CLP"
    PRO_B = "pro-B"
    PRE_B = "pre-B"
    IMMATURE_B = "immature B"
    MATURE_B = "mature B"


STAGE_ORDER = tuple(BCellStage)


@dataclass(frozen=True)
class StageProfile:
    stage: BCellStage
    positive_markers: tuple[str, ...]
    exclusion_markers: tuple[str, ...]
    regulators: tuple[str, ...]
    biological_process: str
    expected_receptor_state: str
    structure_eligible: bool


STAGE_PROFILES: dict[BCellStage, StageProfile] = {
    BCellStage.HSC: StageProfile(
        BCellStage.HSC,
        ("CD34", "KIT", "HLF", "MEIS1", "AVP", "CRHBP", "SPINK2", "GATA2", "MECOM", "HOXA9", "PROM1", "HOPX"),
        ("CD19", "CD79A", "MS4A1", "IGHD"),
        ("RUNX1", "GATA2", "MECOM", "HOXA9"),
        "self-renewal and lymphoid competence",
        "germline IG loci; no assembled BCR expected",
        False,
    ),
    BCellStage.CLP: StageProfile(
        BCellStage.CLP,
        ("CD34", "FLT3", "IL7R", "DNTT", "TCF7", "BCL11A", "TCF3", "IKZF1", "LTB", "SOX4"),
        ("CD3D", "NKG7", "LST1", "MS4A1", "IGHD"),
        ("IKZF1", "TCF3", "BCL11A", "EBF1"),
        "lymphoid commitment before immunoglobulin assembly",
        "germline/early accessibility; no complete BCR expected",
        False,
    ),
    BCellStage.PRO_B: StageProfile(
        BCellStage.PRO_B,
        ("RAG1", "RAG2", "DNTT", "VPREB1", "IGLL1", "EBF1", "PAX5", "CD79A"),
        ("IGHD", "FCER2"),
        ("RAG1", "RAG2", "DNTT", "EBF1", "PAX5"),
        "IGH D-J then V-DJ rearrangement and productive-heavy checkpoint",
        "partial or complete IGH; no conventional light chain required",
        False,
    ),
    BCellStage.PRE_B: StageProfile(
        BCellStage.PRE_B,
        ("VPREB3", "MME", "CD19", "CD79A", "CD79B", "TCL1A", "PAX5"),
        ("IGHD", "FCER2"),
        ("PAX5", "EBF1", "FOXO1", "RAG1", "RAG2"),
        "pre-BCR selection followed by IGK/IGL rearrangement",
        "productive IGH; light-chain rearrangement in progress",
        False,
    ),
    BCellStage.IMMATURE_B: StageProfile(
        BCellStage.IMMATURE_B,
        ("IGHM", "CD19", "CD79A", "CD79B", "CD22", "TCL1A", "HLA-DRA"),
        ("IGHD", "FCER2"),
        ("PAX5", "BACH2", "FOXO1"),
        "paired surface IgM expression, tolerance and receptor editing",
        "productive paired IGH-IGK/IGL expected",
        True,
    ),
    BCellStage.MATURE_B: StageProfile(
        BCellStage.MATURE_B,
        ("MS4A1", "CD19", "CD22", "CD37", "CD74", "HLA-DRA", "CD79A", "CD79B", "IGHD", "FCER2"),
        ("RAG1", "RAG2", "VPREB1", "IGLL1"),
        ("PAX5", "BACH2", "SPIB"),
        "mature BCR signalling and antigen-responsive competence",
        "productive paired IGH-IGK/IGL",
        True,
    ),
}


@dataclass
class VirtualBoneMarrowCell:
    """One virtual cell with stage-aware missingness and provenance.

    A missing receptor or structure means "not observed/not applicable"; it is
    never replaced with a fabricated prediction.
    """

    cell_id: str
    stage: BCellStage
    expression: Mapping[str, float]
    stage_confidence: float
    label_source: str
    paired_receptor: Optional[PairedSequence] = None
    geneformer_state: Optional[RegulatoryState] = None
    alphagenome2_prior: Optional[AlphaGenome2VDJPrior] = None
    structure: Optional[StructureAssessment] = None
    affinity_log10_kd: Optional[float] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def profile(self) -> StageProfile:
        return STAGE_PROFILES[self.stage]

    @property
    def structure_applicable(self) -> bool:
        receptor = self.paired_receptor
        return bool(
            self.profile.structure_eligible and receptor
            and receptor.heavy_aa and receptor.light_aa
        )

    def validate(self) -> None:
        if not 0.0 <= self.stage_confidence <= 1.0:
            raise ValueError("stage_confidence must be in [0, 1]")
        if self.structure is not None and not self.structure_applicable:
            raise ValueError("structure requires an immature/mature cell with paired BCR")
        if self.alphagenome2_prior is not None:
            self.alphagenome2_prior.validate()
            if self.alphagenome2_prior.stage != self.stage.value:
                raise ValueError("AlphaGenome 2 prior stage does not match virtual cell")
        if self.affinity_log10_kd is not None and self.structure is None:
            raise ValueError("affinity labels require an explicit structural/experimental record")


class BoneMarrowTrajectory:
    """Enforce the developmental order without claiming fixed dwell times."""

    @staticmethod
    def next_stage(stage: BCellStage) -> Optional[BCellStage]:
        i = STAGE_ORDER.index(stage)
        return None if i == len(STAGE_ORDER) - 1 else STAGE_ORDER[i + 1]

    @staticmethod
    def can_transition(source: BCellStage, target: BCellStage) -> bool:
        return BoneMarrowTrajectory.next_stage(source) == target

    @staticmethod
    def profiles() -> tuple[StageProfile, ...]:
        return tuple(STAGE_PROFILES[s] for s in STAGE_ORDER)


__all__ = [
    "BCellStage", "STAGE_ORDER", "StageProfile", "STAGE_PROFILES",
    "VirtualBoneMarrowCell", "BoneMarrowTrajectory",
]

