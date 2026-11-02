"""Paired heavy/light B-cell receptor representation and editing history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .affinity import paratope_from
from .regions import ImgtAnnotator
from .vdj import Rearrangement


@dataclass(frozen=True)
class EditingEvent:
    attempt: int
    locus: str
    v_gene: str
    j_gene: str
    reason: str
    productive: bool


@dataclass
class PairedReceptor:
    heavy: Rearrangement
    light: Rearrangement
    editing_history: list[EditingEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.heavy.locus != "IGH":
            raise ValueError("heavy chain must use the IGH locus")
        if self.light.locus not in {"IGK", "IGL"}:
            raise ValueError("light chain must use IGK or IGL")
        if not self.heavy.productive or not self.light.productive:
            raise ValueError("paired receptors require productive chains")

    @property
    def productive(self) -> bool:
        return self.heavy.productive and self.light.productive

    @property
    def sequence_id(self) -> str:
        return f"{self.heavy.v_gene}:{self.heavy.cdr3_aa}|{self.light.v_gene}:{self.light.cdr3_aa}"

    def paratope(self, annotator: Optional[ImgtAnnotator] = None) -> dict[str, str]:
        ann = annotator or ImgtAnnotator()
        hm, lm = ann.annotate(self.heavy), ann.annotate(self.light)
        hp = paratope_from(hm, self.heavy.seq_aa)
        lp = paratope_from(lm, self.light.seq_aa)
        return {
            "h_cdr1": hp["cdr1"], "h_cdr2": hp["cdr2"], "h_cdr3": hp["cdr3"],
            "l_cdr1": lp["cdr1"], "l_cdr2": lp["cdr2"], "l_cdr3": lp["cdr3"],
        }

    def to_dict(self) -> dict:
        return {
            "heavy": self.heavy.to_dict(),
            "light": self.light.to_dict(),
            "editing_history": [asdict(e) for e in self.editing_history],
        }


__all__ = ["EditingEvent", "PairedReceptor"]
