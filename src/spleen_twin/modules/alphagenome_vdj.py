"""AlphaGenome 2 regulatory priors for stage-specific V(D)J rearrangement.

AlphaGenome 2 predicts how reference DNA is read in a selected cell state. It
does not predict a rearranged junction. This module therefore converts
calibrated accessibility, enhancer, chromatin-contact and methylation evidence
into segment-choice weights while OLGA continues to model deletion, P/N
addition, junction sequence and productivity.

The public non-commercial AlphaGenome API terms prohibit using direct model
outputs to train other machine-learning models. Such outputs are accepted here
only for inference-time reranking unless a separate licence permits training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence


class AlphaGenome2Status(str, Enum):
    NOT_CONFIGURED = "not_configured"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    COMPLETE = "complete"


@dataclass(frozen=True)
class GenomicInterval:
    chromosome: str
    start: int
    end: int
    assembly: str = "GRCh38"

    def validate(self) -> None:
        if not self.chromosome or self.start < 0 or self.end <= self.start:
            raise ValueError("invalid half-open genomic interval")


@dataclass(frozen=True)
class SegmentRegulatoryEvidence:
    """Calibrated, unit-interval evidence for one germline segment."""

    segment: str
    locus: str
    interval: GenomicInterval
    accessibility: float
    enhancer_activity: float
    recombination_center_contact: float
    unmethylated_fraction: float
    rss_quality: float
    calibration_id: str

    def validate(self) -> None:
        self.interval.validate()
        if self.locus not in {"IGH", "IGK", "IGL"}:
            raise ValueError("locus must be IGH, IGK or IGL")
        if not self.segment.startswith(self.locus):
            raise ValueError("segment and locus disagree")
        values = (
            self.accessibility, self.enhancer_activity,
            self.recombination_center_contact, self.unmethylated_fraction,
            self.rss_quality,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("regulatory evidence must be calibrated to [0, 1]")
        if not self.calibration_id:
            raise ValueError("calibration_id is required")

    @property
    def regulatory_score(self) -> float:
        # RSS is mechanistic sequence evidence, not an AlphaGenome output.
        return (
            0.30 * self.accessibility
            + 0.20 * self.enhancer_activity
            + 0.25 * self.recombination_center_contact
            + 0.10 * self.unmethylated_fraction
            + 0.15 * self.rss_quality
        )


_ALLOWED_REARRANGEMENT = {
    "pro-B": {"IGH"},
    "pre-B": {"IGK", "IGL"},
    "immature B": {"IGK", "IGL"},  # receptor editing
}


@dataclass(frozen=True)
class AlphaGenome2VDJPrior:
    status: AlphaGenome2Status
    stage: str
    locus: str
    condition_name: str
    model_version: str = "AlphaGenome-2"
    conditioning_cell_count: int = 0
    input_sha256: str = ""
    output_sha256: str = ""
    evidence: tuple[SegmentRegulatoryEvidence, ...] = ()
    usage_scope: str = "noncommercial_research"
    training_use_allowed: bool = False
    error: str = ""

    def validate(self) -> None:
        if self.locus not in {"IGH", "IGK", "IGL"}:
            raise ValueError("invalid immunoglobulin locus")
        if self.conditioning_cell_count < 0 or self.conditioning_cell_count > 256:
            raise ValueError("AlphaGenome 2 accepts at most 256 conditioning cells")
        if self.status is AlphaGenome2Status.COMPLETE:
            if self.locus not in _ALLOWED_REARRANGEMENT.get(self.stage, set()):
                raise ValueError("locus is not rearranged at this developmental stage")
            if not (self.condition_name and self.input_sha256 and self.output_sha256):
                raise ValueError("complete run requires condition and input/output hashes")
            if not self.evidence:
                raise ValueError("complete run requires segment evidence")
            for item in self.evidence:
                item.validate()
                if item.locus != self.locus:
                    raise ValueError("mixed loci in one prior")
        elif self.evidence:
            raise ValueError("non-complete run must not carry model evidence")

    def segment_weights(self, strength: float = 2.0) -> dict[str, float]:
        """Return positive inference-time weights centred around one."""
        self.validate()
        if self.status is not AlphaGenome2Status.COMPLETE:
            return {}
        return {
            item.segment: math.exp(strength * (item.regulatory_score - 0.5))
            for item in self.evidence
        }

    def as_training_features(self) -> Mapping[str, float]:
        """Block training use unless a separately recorded licence permits it."""
        if not self.training_use_allowed:
            raise PermissionError(
                "AlphaGenome API outputs may not be used to train another model "
                "under the non-commercial API terms"
            )
        return {item.segment: item.regulatory_score for item in self.evidence}


class AlphaGenome2Backend(Protocol):
    """Contract for the official API or a separately licensed endpoint."""

    def build_prior(
        self,
        *,
        stage: str,
        locus: str,
        intervals: Sequence[GenomicInterval],
        raw_umi_counts,
        condition_name: str,
    ) -> AlphaGenome2VDJPrior: ...


class UnavailableAlphaGenome2Backend:
    """Explicit no-credentials fallback; never fabricates tracks."""

    def __init__(self, reason: str = "AlphaGenome 2 API key/backend not configured") -> None:
        self.reason = reason

    def build_prior(self, *, stage: str, locus: str, intervals,
                    raw_umi_counts, condition_name: str) -> AlphaGenome2VDJPrior:
        return AlphaGenome2VDJPrior(
            status=AlphaGenome2Status.UNAVAILABLE,
            stage=stage,
            locus=locus,
            condition_name=condition_name,
            error=self.reason,
        )


__all__ = [
    "AlphaGenome2Status", "GenomicInterval", "SegmentRegulatoryEvidence",
    "AlphaGenome2VDJPrior", "AlphaGenome2Backend",
    "UnavailableAlphaGenome2Backend",
]
