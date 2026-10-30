"""Train — fit a sequence-to-property model on curated records.

The model is deliberately simple and fully inspectable: composition, k-mer
and positional features over the CDR loops, with ridge regression on
log10 KD.  Its job in this repository is to be an honest, reproducible
baseline inside the loop, not to compete with a protein language model.

The split is **by clonal lineage, not by sequence**.  Sequences from one
germinal-centre lineage differ by a handful of mutations; splitting them
randomly puts near-duplicates on both sides of the split and inflates
every metric.  This is among the most common ways an antibody ML result
fails to replicate, and it is why the loop reports the number of
lineages alongside the number of sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def build_feature_spec(records, k: int = 2, max_cdr3: int = 30) -> dict:
    """Fix the feature vocabulary on the training set, once."""
    kmers: dict[str, int] = {}
    for r in records:
        s = r.cdr1_aa + r.cdr2_aa + r.cdr3_aa
        for i in range(len(s) - k + 1):
            kmers.setdefault(s[i:i + k], len(kmers))
    return {"kmers": kmers, "k": k, "max_cdr3": max_cdr3}


def featurise(records, spec: dict) -> np.ndarray:
    """Encode records against a fixed feature vocabulary."""
    aa_index = {a: i for i, a in enumerate(AMINO_ACIDS)}
    kmers, k, max_cdr3 = spec["kmers"], spec["k"], spec["max_cdr3"]
    n_comp, n_kmer = len(AMINO_ACIDS), len(kmers)
    n_pos = max_cdr3 * len(AMINO_ACIDS)
    X = np.zeros((len(records), n_comp + n_kmer + n_pos + 2), dtype=np.float32)

    for row, r in enumerate(records):
        s = r.cdr1_aa + r.cdr2_aa + r.cdr3_aa
        for a in s:
            if a in aa_index:
                X[row, aa_index[a]] += 1.0
        if s:
            X[row, :n_comp] /= len(s)
        for i in range(len(s) - k + 1):
            j = kmers.get(s[i:i + k])
            if j is not None:
                X[row, n_comp + j] += 1.0
        for p, a in enumerate(r.cdr3_aa[:max_cdr3]):
            if a in aa_index:
                X[row, n_comp + n_kmer + p * len(AMINO_ACIDS) + aa_index[a]] = 1.0
        X[row, -2] = len(r.cdr3_aa)
        X[row, -1] = r.n_v_mutations
    return X


def lineage_key(record) -> str:
    """Proxy for clonal lineage: V gene, CDR3 length and junction prefix."""
    return f"{record.v_call}|{len(record.junction_aa)}|{record.junction_aa[:6]}"


def lineage_split(records, test_fraction: float = 0.25, seed: int = 0):
    """Hold out whole lineages, never individual sequences."""
    rng = np.random.default_rng(seed)
    keys = [lineage_key(r) for r in records]
    uniq = sorted(set(keys))
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_fraction))
    test_keys = set(uniq[:n_test])
    train_idx = [i for i, kk in enumerate(keys) if kk not in test_keys]
    test_idx = [i for i, kk in enumerate(keys) if kk in test_keys]
    return train_idx, test_idx, len(uniq)


@dataclass
class TrainedModel:
    kind: str
    coef: np.ndarray
    intercept: float
    feature_spec: dict
    metrics: dict = field(default_factory=dict)

    def predict(self, records) -> np.ndarray:
        X = featurise(records, self.feature_spec)
        return X @ self.coef + self.intercept


def train_affinity_model(records, alpha: float = 1.0, test_fraction: float = 0.25,
                         seed: int = 0) -> TrainedModel:
    """Ridge regression on log10 KD with a lineage-aware split."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.stats import spearmanr

    usable = [r for r in records if r.log10_kd is not None]
    if len(usable) < 20:
        raise ValueError(f"only {len(usable)} labelled records; need at least 20")

    tr, te, n_lineages = lineage_split(usable, test_fraction, seed)
    spec = build_feature_spec([usable[i] for i in tr])
    X = featurise(usable, spec)
    y = np.array([r.log10_kd for r in usable], dtype=float)

    model = Ridge(alpha=alpha)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    rho = float(spearmanr(y[te], pred).statistic) if len(te) > 2 else float("nan")

    return TrainedModel(
        kind="ridge_log10_kd",
        coef=model.coef_,
        intercept=float(model.intercept_),
        feature_spec=spec,
        metrics={
            "n_train": len(tr), "n_test": len(te), "n_lineages": n_lineages,
            "test_mae_log10": float(mean_absolute_error(y[te], pred)),
            "test_r2": float(r2_score(y[te], pred)) if len(te) > 2 else float("nan"),
            "test_spearman": rho,
            "split": "lineage-aware (V gene + CDR3 length + junction prefix)",
        },
    )


__all__ = ["train_affinity_model", "TrainedModel", "featurise",
           "build_feature_spec", "lineage_split", "lineage_key", "AMINO_ACIDS"]
