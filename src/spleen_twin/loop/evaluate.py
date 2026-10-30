"""Evaluate — did the model beat the baseline, and what failed?

Two things are recorded every round:

* **Model versus baseline.**  The model-guided arm is compared with the
  random arm on the same parents and the same measurement.  If the model
  does not win, the loop records that it did not.
* **Failures.**  Every proposal that lost affinity, failed developability,
  or was predicted well and measured badly is written to a failure log.
  These are the loop's most valuable records and the ones most often
  thrown away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RoundEvaluation:
    round_index: int
    n_model: int
    n_baseline: int
    best_model_log10_kd: float
    best_baseline_log10_kd: float
    median_model_log10_kd: float
    median_baseline_log10_kd: float
    model_beats_baseline: bool
    model_margin_log10: float
    hit_rate_model: float
    hit_rate_baseline: float
    prediction_spearman: Optional[float] = None
    n_failures_logged: int = 0
    notes: str = ""


def evaluate_round(round_index: int, measured, parent_best_log10_kd: float,
                   hit_threshold_log10_kd: float = -8.5) -> RoundEvaluation:
    """Compare the model arm with the baseline arm on identical footing."""
    from scipy.stats import spearmanr

    model = [r for r in measured if r.metadata.get("proposer") == "model_guided"]
    base = [r for r in measured if r.metadata.get("proposer") == "random_baseline"]

    def stats(rs):
        if not rs:
            return float("nan"), float("nan"), 0.0
        v = np.array([r.log10_kd for r in rs])
        return float(v.min()), float(np.median(v)), float((v <= hit_threshold_log10_kd).mean())

    bm, mm, hm = stats(model)
    bb, mb, hb = stats(base)

    rho = None
    preds = [(r.metadata.get("predicted_log10_kd"), r.log10_kd)
             for r in model if r.metadata.get("predicted_log10_kd") is not None]
    if len(preds) > 2:
        p, a = zip(*preds)
        rho = float(spearmanr(p, a).statistic)

    wins = bool(bm < bb)
    return RoundEvaluation(
        round_index=round_index, n_model=len(model), n_baseline=len(base),
        best_model_log10_kd=bm, best_baseline_log10_kd=bb,
        median_model_log10_kd=mm, median_baseline_log10_kd=mb,
        model_beats_baseline=wins,
        model_margin_log10=float(bb - bm) if np.isfinite(bm) and np.isfinite(bb) else float("nan"),
        hit_rate_model=hm, hit_rate_baseline=hb,
        prediction_spearman=rho,
        notes="" if wins else "model arm did not beat the random baseline this round",
    )


def log_failures(measured, parent_best_log10_kd: float, path: str,
                 regression_margin: float = 0.0) -> int:
    """Write every proposal that regressed, failed a gate, or was mispredicted."""
    failures = []
    for r in measured:
        reasons = []
        if r.log10_kd is not None and r.log10_kd > parent_best_log10_kd + regression_margin:
            reasons.append("affinity regressed versus parent")
        if r.developability_tier == "C":
            reasons.append("developability tier C")
        pred = r.metadata.get("predicted_log10_kd")
        if pred is not None and r.log10_kd is not None and (r.log10_kd - pred) > 1.0:
            reasons.append(f"overpredicted by {r.log10_kd - pred:.2f} log units")
        if reasons:
            failures.append({
                "sequence_id": r.sequence_id,
                "proposer": r.metadata.get("proposer"),
                "log10_kd": r.log10_kd, "predicted_log10_kd": pred,
                "developability_tier": r.developability_tier,
                "reasons": reasons,
            })
    with open(path, "w") as fh:
        json.dump(failures, fh, indent=1)
    return len(failures)


__all__ = ["evaluate_round", "RoundEvaluation", "log_failures"]
