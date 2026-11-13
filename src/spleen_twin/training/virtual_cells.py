"""Build GeneFormer-X/VDJ/structure-ready virtual-cell training examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

from ..organ.bone_marrow import BCellStage, STAGE_PROFILES, VirtualBoneMarrowCell


def _detected(expression: Mapping[str, float], genes: Sequence[str]) -> int:
    return sum(float(expression.get(g, 0.0)) > 0.0 for g in genes)


class MarkerStageAnnotator:
    """Conservative marker rules for pseudo-labelling the local 10x marrow set.

    These labels are training hypotheses, not reference annotations. Cells that
    do not satisfy a specific rule return ``None`` and are not forced onto the
    developmental trajectory.
    """

    label_source = "marker_heuristic_v1"

    def classify(self, expression: Mapping[str, float]) -> tuple[Optional[BCellStage], float]:
        d = lambda *genes: _detected(expression, genes)

        hsc_program = ("HLF", "CRHBP", "SPINK2", "GATA2", "MECOM", "HOXA9",
                       "MEIS1", "KIT", "PROM1", "HOPX")
        if (d("CD34") and d(*hsc_program) >= 3
                and d("CD3D", "NKG7", "LYZ", "MPO", "CD79A", "DNTT", "VPREB1") == 0):
            return BCellStage.HSC, min(1.0, 0.75 + 0.04 * d(*hsc_program))
        if (d("CD34") and d("DNTT", "VPREB1") >= 1
                and d("CD79A", "MME", "IGLL1") >= 1
                and d("CD3D", "NKG7", "LYZ", "MPO") == 0):
            return BCellStage.PRO_B, 0.85
        if (d("IGHM") >= 1
                and d("CD79A", "CD79B", "CD19", "MS4A1", "CD22") >= 2
                and d("CD24", "TCL1A") >= 1
                and d("IGHD", "FCER2") == 0):
            return BCellStage.IMMATURE_B, 0.86
        if (d("IGHD", "FCER2") >= 1
                and d("MS4A1") >= 1
                and d("CD79A", "CD79B", "CD19", "MS4A1", "CD22") >= 2):
            return BCellStage.MATURE_B, 0.90
        if (d("VPREB3", "MME") >= 1
                and d("CD79A", "CD79B", "CD19", "PAX5", "TCL1A") >= 2
                and d("IGHD", "FCER2") == 0):
            return BCellStage.PRE_B, 0.82
        if (d("CD34") and d("FLT3") and d("IL7R", "TCF7", "DNTT") >= 1
                and d("CD79A", "CD19", "MS4A1") == 0
                and d("CD3D", "NKG7", "LYZ", "MPO") == 0):
            return BCellStage.CLP, 0.78
        return None, 0.0


@dataclass(frozen=True)
class VirtualCellTrainingExample:
    cell_id: str
    stage: str
    stage_confidence: float
    label_source: str
    gene_tokens: tuple[str, ...]
    gene_values: tuple[float, ...]
    regulatory_edges: tuple[tuple[str, str], ...]
    heavy_aa: str = ""
    light_aa: str = ""
    heavy_v: str = ""
    heavy_d: str = ""
    heavy_j: str = ""
    light_v: str = ""
    light_j: str = ""
    structure_available: bool = False
    affinity_available: bool = False
    loss_mask: Mapping[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def make_training_example(
    cell: VirtualBoneMarrowCell,
    regulatory_edges: Sequence[tuple[str, str]],
    max_genes: int = 2048,
) -> VirtualCellTrainingExample:
    cell.validate()
    ranked = sorted(
        ((g, float(v)) for g, v in cell.expression.items() if float(v) > 0),
        key=lambda item: (-item[1], item[0]),
    )[:max_genes]
    receptor = cell.paired_receptor
    return VirtualCellTrainingExample(
        cell_id=cell.cell_id,
        stage=cell.stage.value,
        stage_confidence=cell.stage_confidence,
        label_source=cell.label_source,
        gene_tokens=tuple(g for g, _ in ranked),
        gene_values=tuple(v for _, v in ranked),
        regulatory_edges=tuple(regulatory_edges),
        heavy_aa=receptor.heavy_aa if receptor else "",
        light_aa=receptor.light_aa if receptor else "",
        heavy_v=receptor.heavy_v if receptor else "",
        heavy_d=receptor.heavy_d if receptor else "",
        heavy_j=receptor.heavy_j if receptor else "",
        light_v=receptor.light_v if receptor else "",
        light_j=receptor.light_j if receptor else "",
        structure_available=cell.structure is not None,
        affinity_available=cell.affinity_log10_kd is not None,
        loss_mask={
            "cell_state": True,
            "vdj_heavy": bool(receptor and receptor.heavy_aa),
            "vdj_light": bool(receptor and receptor.light_aa),
            "heavy_light_pair": bool(receptor and receptor.heavy_aa and receptor.light_aa),
            "structure": cell.structure is not None,
            "affinity": cell.affinity_log10_kd is not None,
        },
    )


def stage_regulatory_edges(stage: BCellStage) -> tuple[tuple[str, str], ...]:
    regulators = STAGE_PROFILES[stage].regulators
    markers = STAGE_PROFILES[stage].positive_markers
    return tuple((reg, marker) for reg in regulators for marker in markers if reg != marker)


__all__ = [
    "MarkerStageAnnotator", "VirtualCellTrainingExample",
    "make_training_example", "stage_regulatory_edges",
]

