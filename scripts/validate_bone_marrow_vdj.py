#!/usr/bin/env python3
"""Validate the compact 10x bone-marrow GEX/ADT + BCR V(D)J core."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def decode(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix_path = next(args.data_dir.glob("*_count_sample_feature_bc_matrix.h5"))
    contig_path = next(args.data_dir.glob("*_vdj_b_filtered_contig_annotations.csv"))
    clonotype_path = next(args.data_dir.glob("*_vdj_b_clonotypes.csv"))

    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("h5py is required to validate the 10x matrix") from exc

    with h5py.File(matrix_path, "r") as handle:
        matrix = handle["matrix"]
        matrix_shape = [int(v) for v in matrix["shape"][:]]
        matrix_barcodes = {decode(v) for v in matrix["barcodes"][:]}
        feature_types = Counter(decode(v) for v in matrix["features"]["feature_type"][:])

    chains_by_barcode: dict[str, set[str]] = defaultdict(set)
    productive_chains_by_barcode: dict[str, set[str]] = defaultdict(set)
    chain_counts: Counter[str] = Counter()
    productive_chain_counts: Counter[str] = Counter()
    with contig_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        barcode = row["barcode"]
        chain = row["chain"]
        chains_by_barcode[barcode].add(chain)
        chain_counts[chain] += 1
        if row["productive"].lower() == "true":
            productive_chains_by_barcode[barcode].add(chain)
            productive_chain_counts[chain] += 1

    paired = {
        barcode
        for barcode, chains in productive_chains_by_barcode.items()
        if "IGH" in chains and bool({"IGK", "IGL"} & chains)
    }
    with clonotype_path.open(newline="") as stream:
        clonotypes = sum(1 for _ in csv.DictReader(stream))

    vdj_barcodes = set(chains_by_barcode)
    summary = {
        "matrix": {
            "features_by_barcodes": matrix_shape,
            "feature_types": dict(sorted(feature_types.items())),
            "barcodes": len(matrix_barcodes),
        },
        "bcr_vdj": {
            "contigs": len(rows),
            "cells_with_bcr_contigs": len(vdj_barcodes),
            "cells_with_productive_paired_igh_and_light_chain": len(paired),
            "chain_counts": dict(sorted(chain_counts.items())),
            "productive_chain_counts": dict(sorted(productive_chain_counts.items())),
            "clonotypes": clonotypes,
            "bcr_barcodes_present_in_expression_matrix": len(vdj_barcodes & matrix_barcodes),
            "bcr_barcodes_missing_from_expression_matrix": len(vdj_barcodes - matrix_barcodes),
        },
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

