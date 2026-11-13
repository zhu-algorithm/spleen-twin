#!/usr/bin/env python3
"""Train a transparent pilot stage head on frozen official Geneformer V1 embeddings.

This is deliberately not described as a validated six-stage foundation model:
the local dataset has very few early B-lineage cells and marker-derived labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import h5py
import joblib
import numpy as np
import torch
from scipy.sparse import csc_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import BertModel


REGULATORY_GENES = (
    "RUNX1", "GATA2", "MECOM", "HOXA9", "IKZF1", "TCF3", "BCL11A",
    "EBF1", "PAX5", "RAG1", "RAG2", "DNTT", "FOXO1", "BACH2",
    "SPIB", "VPREB1", "IGLL1", "AICDA",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_labels(path: Path, max_per_stage: int) -> list[tuple[str, str]]:
    by_stage: dict[str, list[str]] = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            barcode = row["cell_id"].rsplit(":", 1)[-1]
            by_stage.setdefault(row["stage"], []).append(barcode)
    selected = []
    for stage in sorted(by_stage):
        for barcode in sorted(by_stage[stage])[:max_per_stage]:
            selected.append((barcode, stage))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/bone_marrow/10x_20k_bmmnc_5p_ht_v2"))
    parser.add_argument("--model-dir", type=Path, default=Path("data/external/models/geneformer-v1-10m"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/models/geneformer-stage-pilot"))
    parser.add_argument("--summary", type=Path, default=Path("data/geneformer_stage_pilot_summary.json"))
    parser.add_argument("--max-per-stage", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_path = args.input_dir / "virtual_cell_training.jsonl"
    matrix_path = next(args.input_dir.glob("*_feature_bc_matrix.h5"))
    selected = load_labels(training_path, args.max_per_stage)
    label_by_barcode = dict(selected)

    with (args.model_dir / "token_dictionary_gc30M.pkl").open("rb") as stream:
        token_dict = pickle.load(stream)
    with (args.model_dir / "gene_median_dictionary_gc30M.pkl").open("rb") as stream:
        medians = pickle.load(stream)

    token_sequences, regulator_rows, labels, cell_ids = [], [], [], []
    with h5py.File(matrix_path, "r") as handle:
        matrix = handle["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][:])
        x = csc_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape)
        ids = [v.decode() for v in matrix["features"]["id"][:]]
        names = [v.decode() for v in matrix["features"]["name"][:]]
        types = [v.decode() for v in matrix["features"]["feature_type"][:]]
        barcodes = [v.decode() for v in matrix["barcodes"][:]]
        name_to_row = {name: i for i, (name, typ) in enumerate(zip(names, types)) if typ == "Gene Expression"}

        for col, barcode in enumerate(barcodes):
            if barcode not in label_by_barcode:
                continue
            start, stop = x.indptr[col], x.indptr[col + 1]
            rows = x.indices[start:stop]
            counts = x.data[start:stop].astype(float)
            keep = np.array([types[int(row)] == "Gene Expression" for row in rows])
            rows, counts = rows[keep], counts[keep]
            total = float(counts.sum()) or 1.0
            ranked = []
            for row, count in zip(rows, counts):
                gene_id = ids[int(row)].split(".", 1)[0]
                if gene_id in token_dict and gene_id in medians and medians[gene_id] > 0:
                    value = count / total * 10_000.0 / float(medians[gene_id])
                    ranked.append((value, int(token_dict[gene_id])))
            ranked.sort(reverse=True)
            token_sequences.append([token for _, token in ranked[:args.max_length]])
            regulator_rows.append([
                np.log1p(float(x[name_to_row[g], col]) / total * 10_000.0)
                if g in name_to_row else 0.0
                for g in REGULATORY_GENES
            ])
            labels.append(label_by_barcode[barcode])
            cell_ids.append(barcode)

    model = BertModel.from_pretrained(args.model_dir, local_files_only=True)
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(token_sequences), args.batch_size):
            batch = token_sequences[start:start + args.batch_size]
            width = max(len(row) for row in batch)
            tokens = torch.zeros((len(batch), width), dtype=torch.long)
            attention = torch.zeros_like(tokens)
            for i, row in enumerate(batch):
                tokens[i, :len(row)] = torch.tensor(row, dtype=torch.long)
                attention[i, :len(row)] = 1
            hidden = model(input_ids=tokens, attention_mask=attention).last_hidden_state
            pooled = (hidden * attention.unsqueeze(-1)).sum(1) / attention.sum(1, keepdim=True).clamp_min(1)
            embeddings.append(pooled.cpu().numpy())
    geneformer_embeddings = np.concatenate(embeddings)
    features = np.concatenate([geneformer_embeddings, np.asarray(regulator_rows)], axis=1)
    labels_array = np.asarray(labels)

    head = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs", C=0.25),
    )
    head.fit(features, labels_array)
    predicted = head.predict(features)

    head_path = args.output_dir / "stage_head.joblib"
    embeddings_path = args.output_dir / "training_embeddings.npz"
    joblib.dump(head, head_path)
    np.savez_compressed(
        embeddings_path,
        cell_ids=np.asarray(cell_ids), labels=labels_array,
        geneformer_embeddings=geneformer_embeddings,
        regulatory_features=np.asarray(regulator_rows),
        regulatory_genes=np.asarray(REGULATORY_GENES),
    )
    summary = {
        "schema_version": "geneformer-stage-pilot-1",
        "status": "trained_pilot_not_validated",
        "base_model": "ctheodoris/Geneformer Geneformer-V1-10M",
        "base_model_license": "Apache-2.0",
        "base_model_sha256": file_sha256(args.model_dir / "model.safetensors"),
        "training_corpus_sha256": file_sha256(training_path),
        "cells_used": len(labels),
        "class_counts": dict(Counter(labels)),
        "embedding_dimensions": int(geneformer_embeddings.shape[1]),
        "regulatory_features": list(REGULATORY_GENES),
        "max_sequence_length": args.max_length,
        "head": "standardized balanced multinomial logistic regression",
        "metrics": {
            "kind": "resubstitution_only_not_generalization",
            "balanced_accuracy": float(balanced_accuracy_score(labels_array, predicted)),
            "macro_f1": float(f1_score(labels_array, predicted, average="macro")),
        },
        "artifacts": {
            "stage_head": str(head_path),
            "stage_head_sha256": file_sha256(head_path),
            "embeddings": str(embeddings_path),
            "embeddings_sha256": file_sha256(embeddings_path),
        },
        "limitations": [
            "Stage labels are marker-derived pseudo-labels.",
            "Rare early stages are too small for held-out six-class evaluation.",
            "Metrics are training-set fit diagnostics and must not be reported as predictive performance.",
            "This pilot freezes Geneformer and trains only a regulatory-aware stage head.",
            "No AlphaFold structure was generated or used as an affinity label.",
        ],
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

