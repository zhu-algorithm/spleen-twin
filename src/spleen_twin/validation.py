"""Executable distributional checks for M0/M1 acceptance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .modules.vdj import ChainRecombinator


@dataclass(frozen=True)
class DistributionCheck:
    name: str
    value: float
    lower: float
    upper: float
    passed: bool
    evidence: str


def validate_vdj(seed: int = 0, n: int = 4000, locus: str = "IGH") -> dict:
    """Run versioned biological invariants against a seeded repertoire."""
    rc = ChainRecombinator(locus, rng=np.random.default_rng(seed))
    sample = [r for r in (rc.rearrange() for _ in range(n)) if r is not None]
    productive = [r for r in sample if r.productive]
    in_frame = sum(r.in_frame for r in sample) / max(1, len(sample))
    productive_fraction = len(productive) / max(1, len(sample))
    mean_cdr3 = float(np.mean([r.cdr3_len_aa - 2 for r in productive]))
    specs = [
        ("in_frame_fraction", in_frame, 0.28, 0.40,
         "frame-agnostic junctions; expected near one third"),
        ("productive_fraction", productive_fraction, 0.12, 0.32,
         "productive subset after frame, stop and conserved-anchor checks"),
        ("mean_cdr3_aa_excluding_anchors", mean_cdr3, 12.0, 19.0,
         "broad human CDR-H3 plausibility interval"),
    ]
    checks = [DistributionCheck(nm, v, lo, hi, lo < v < hi, ev) for nm, v, lo, hi, ev in specs]
    return {
        "schema_version": "m0-m1-validation-1", "seed": seed, "n_requested": n,
        "n_generated": len(sample), "locus": locus,
        "checks": [asdict(c) for c in checks],
        "passed": all(c.passed for c in checks),
    }


__all__ = ["DistributionCheck", "validate_vdj"]
