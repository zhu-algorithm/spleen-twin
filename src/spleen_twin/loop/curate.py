"""Curate — the step the closed loop actually breaks at.

The limiting resource in AI antibody design is not model capacity, it is
labelled, comparable, provenance-complete data.  Public structure-affinity
data is roughly three orders of magnitude smaller than the sequence data
models are pretrained on, and campaign feedback is usually recorded in a
form that cannot be learned from.  Three rules make a record learnable, and
this module enforces all three:

1. **One experimental system per label.**  A KD is only comparable to
   another KD measured the same way, so the assay and its conditions are
   part of the label, not a footnote.
2. **Complete metadata.**  Animal, immunisation schedule, antigen lot and
   conformation, assay conditions, fitting assumptions and a pointer to
   the raw record.  A record missing any required field is *rejected*,
   not silently imputed.
3. **Negatives are kept.**  Non-binders, poor expressers and failed
   developability are the labels that teach a model where the cliff is.
   Dropping them is the single most common way a loop stops compounding.

The record layout follows AIRR Rearrangement conventions where they apply
(``sequence``, ``v_call``, ``d_call``, ``j_call``, ``junction_aa``,
``productive``) so curated output can be read by existing repertoire tools.

    Vander Heiden JA, et al. "AIRR Community Standardized Representations
    for Annotated Immune Repertoires." Front Immunol 9:2206 (2018).
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Optional

#: Metadata fields without which a measurement is not learnable.
REQUIRED_METADATA = (
    "campaign_id",
    "subject_id",
    "immunisation_schedule",
    "antigen_id",
    "antigen_lot",
    "antigen_conformation",
    "assay",
    "assay_conditions",
    "fitting_model",
    "raw_record_uri",
)

#: Assay families whose numbers may not be pooled with one another.
ASSAY_FAMILIES = {
    "SPR": "kinetic", "BLI": "kinetic", "KinExA": "equilibrium",
    "ELISA": "equilibrium", "FACS_titration": "equilibrium",
    "simulated": "simulated",
}


@dataclass
class AntibodyRecord:
    """One curated, learnable observation."""

    # --- identity (AIRR-aligned) ---
    sequence_id: str
    sequence: str
    sequence_aa: str
    v_call: str
    d_call: Optional[str]
    j_call: str
    junction_aa: str
    productive: bool
    cdr1_aa: str = ""
    cdr2_aa: str = ""
    cdr3_aa: str = ""
    n_v_mutations: int = 0

    # --- labels ---
    kd_molar: Optional[float] = None
    log10_kd: Optional[float] = None
    binder: Optional[bool] = None
    expression_g_l: Optional[float] = None
    tm_celsius: Optional[float] = None
    developability_tier: Optional[str] = None

    # --- provenance ---
    metadata: dict = field(default_factory=dict)
    curated_at: str = ""
    label_source: str = "simulated"

    def checksum(self) -> str:
        """Identity of the molecule, for deduplication.

        Designed variants carry no nucleotide sequence, so the amino-acid
        sequence is the fallback identity; hashing an empty nucleotide
        field would collapse every design onto one checksum.
        """
        basis = self.sequence or self.sequence_aa
        if not basis:
            basis = f"{self.v_call}|{self.cdr1_aa}|{self.cdr2_aa}|{self.cdr3_aa}"
        return hashlib.sha256(basis.encode()).hexdigest()[:16]


class CurationError(ValueError):
    """Raised when a record cannot be made learnable."""


class Curator:
    """Validates, deduplicates and writes learnable records."""

    def __init__(self, required_metadata: Iterable[str] = REQUIRED_METADATA,
                 keep_negatives: bool = True) -> None:
        self.required = tuple(required_metadata)
        self.keep_negatives = keep_negatives
        self.records: list[AntibodyRecord] = []
        self.rejected: list[dict] = []
        self._seen: set[str] = set()

    def validate(self, record: AntibodyRecord) -> list[str]:
        problems = []
        missing = [f for f in self.required if not record.metadata.get(f)]
        if missing:
            problems.append(f"missing metadata: {','.join(missing)}")
        assay = record.metadata.get("assay")
        if assay and assay not in ASSAY_FAMILIES:
            problems.append(f"unknown assay {assay!r}")
        if record.kd_molar is None and record.binder is None:
            problems.append("record carries no label at all")
        if record.kd_molar is not None and record.kd_molar <= 0:
            problems.append("non-positive KD")
        if not record.sequence_aa:
            problems.append("empty amino-acid sequence")
        return problems

    def add(self, record: AntibodyRecord) -> bool:
        """Add one record.  Returns False if it was rejected."""
        problems = self.validate(record)
        if problems:
            self.rejected.append({"sequence_id": record.sequence_id, "problems": problems})
            return False
        if not self.keep_negatives and record.binder is False:
            self.rejected.append({"sequence_id": record.sequence_id,
                                  "problems": ["negative dropped by configuration"]})
            return False
        key = record.checksum()
        if key in self._seen:
            self.rejected.append({"sequence_id": record.sequence_id,
                                  "problems": ["duplicate sequence"]})
            return False
        self._seen.add(key)
        record.curated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.records.append(record)
        return True

    # -- reporting ----------------------------------------------------------
    def summary(self) -> dict:
        n = len(self.records)
        binders = sum(1 for r in self.records if r.binder)
        assays = {}
        for r in self.records:
            a = r.metadata.get("assay", "unknown")
            assays[a] = assays.get(a, 0) + 1
        return {
            "n_curated": n,
            "n_rejected": len(self.rejected),
            "n_binders": binders,
            "n_non_binders": n - binders,
            "negative_fraction": (n - binders) / n if n else 0.0,
            "assays": assays,
            "comparable": len({ASSAY_FAMILIES.get(a, "unknown") for a in assays}) <= 1,
        }

    def to_airr_tsv(self, path: str) -> None:
        """Write an AIRR-style rearrangement table."""
        cols = ["sequence_id", "sequence", "sequence_aa", "v_call", "d_call",
                "j_call", "junction_aa", "productive", "cdr1_aa", "cdr2_aa",
                "cdr3_aa", "n_v_mutations", "kd_molar", "log10_kd", "binder",
                "expression_g_l", "tm_celsius", "developability_tier",
                "label_source", "curated_at"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
            w.writeheader()
            for r in self.records:
                w.writerow(asdict(r))

    def to_jsonl(self, path: str) -> None:
        """Write records with their full metadata, one JSON object per line."""
        with open(path, "w") as fh:
            for r in self.records:
                fh.write(json.dumps(asdict(r)) + "\n")

    def write_rejections(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.rejected, fh, indent=1)


__all__ = ["Curator", "AntibodyRecord", "CurationError",
           "REQUIRED_METADATA", "ASSAY_FAMILIES"]
