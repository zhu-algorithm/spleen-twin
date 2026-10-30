"""Design — propose new variants for the next round.

Two proposers are provided and both are always run, because a model is
only informative relative to a baseline that does not use it.  In the
AIntibody blind benchmark a non-AI consensus baseline outscored all but
two of 29 participating groups, and on the ranking task most models did
worse than random selection; a loop that never runs the baseline cannot
tell whether its model is contributing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .train import AMINO_ACIDS, TrainedModel


@dataclass
class Proposal:
    parent_id: str
    sequence_aa: str
    cdr1_aa: str
    cdr2_aa: str
    cdr3_aa: str
    v_call: str
    junction_aa: str
    n_v_mutations: int
    proposer: str
    predicted_log10_kd: Optional[float] = None


class _Shim:
    """Minimal record-like object so proposals can be featurised."""

    def __init__(self, p: "Proposal"):
        self.cdr1_aa, self.cdr2_aa, self.cdr3_aa = p.cdr1_aa, p.cdr2_aa, p.cdr3_aa
        self.junction_aa, self.v_call = p.junction_aa, p.v_call
        self.n_v_mutations = p.n_v_mutations


def _mutate_cdr3(cdr3: str, n: int, rng) -> str:
    """Substitute interior CDR3 residues, leaving the anchors intact."""
    if len(cdr3) <= 4:
        return cdr3
    chars = list(cdr3)
    n = min(n, len(chars) - 2)
    positions = rng.choice(np.arange(1, len(chars) - 1), size=n, replace=False)
    for p in positions:
        chars[int(p)] = AMINO_ACIDS[int(rng.integers(len(AMINO_ACIDS)))]
    return "".join(chars)


def propose_random(parents, n_per_parent: int = 8, n_mutations: int = 2,
                   seed: int = 0) -> list[Proposal]:
    """Baseline proposer: random CDR3 substitutions, no model involved."""
    rng = np.random.default_rng(seed)
    out: list[Proposal] = []
    for r in parents:
        for _ in range(n_per_parent):
            new_cdr3 = _mutate_cdr3(r.cdr3_aa, n_mutations, rng)
            seq = r.sequence_aa.replace(r.cdr3_aa, new_cdr3) if r.cdr3_aa else r.sequence_aa
            out.append(Proposal(
                parent_id=r.sequence_id, sequence_aa=seq,
                cdr1_aa=r.cdr1_aa, cdr2_aa=r.cdr2_aa, cdr3_aa=new_cdr3,
                v_call=r.v_call, junction_aa=r.junction_aa,
                n_v_mutations=r.n_v_mutations + n_mutations,
                proposer="random_baseline",
            ))
    return out


def propose_model_guided(parents, model: TrainedModel, n_per_parent: int = 8,
                         n_candidates: int = 64, n_mutations: int = 2,
                         seed: int = 0) -> list[Proposal]:
    """Model proposer: oversample variants, keep the best-predicted."""
    rng = np.random.default_rng(seed)
    out: list[Proposal] = []
    for r in parents:
        pool = propose_random([r], n_per_parent=n_candidates,
                              n_mutations=n_mutations,
                              seed=int(rng.integers(1 << 30)))
        preds = model.predict([_Shim(p) for p in pool])
        for p, v in zip(pool, preds):
            p.predicted_log10_kd = float(v)
            p.proposer = "model_guided"
        pool.sort(key=lambda p: p.predicted_log10_kd)
        out.extend(pool[:n_per_parent])
    return out


__all__ = ["propose_random", "propose_model_guided", "Proposal"]
