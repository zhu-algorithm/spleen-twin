#!/usr/bin/env python3
"""Build a stage-aware GeneFormer-X/VDJ training corpus from local 10x BMMNC data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
from scipy.sparse import csc_matrix

from spleen_twin.modules.ai_maturation import PairedSequence
from spleen_twin.organ.bone_marrow import VirtualBoneMarrowCell
from spleen_twin.training.virtual_cells import (
    MarkerStageAnnotator, make_training_example, stage_regulatory_edges,
)


AA_FIELDS = ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_receptors(path: Path) -> dict[str, PairedSequence]:
    chains: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            chains[row["barcode"]][row["chain"]].append(row)

    def best(rows: list[dict]) -> dict | None:
        return max(rows, key=lambda row: (int(row.get("umis") or 0), int(row.get("reads") or 0))) if rows else None

    out = {}
    for barcode, by_chain in chains.items():
        heavy = best(by_chain.get("IGH", []))
        light = best(by_chain.get("IGK", []) + by_chain.get("IGL", []))
        if heavy is None:
            continue
        out[barcode] = PairedSequence(
            heavy_aa="".join(heavy.get(field, "") for field in AA_FIELDS),
            light_aa="".join(light.get(field, "") for field in AA_FIELDS) if light else "",
            heavy_v=heavy.get("v_gene", ""),
            heavy_d=heavy.get("d_gene", ""),
            heavy_j=heavy.get("j_gene", ""),
            light_v=light.get("v_gene", "") if light else "",
            light_j=light.get("j_gene", "") if light else "",
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external/bone_marrow/10x_20k_bmmnc_5p_ht_v2"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=Path("data/bone_marrow_training_summary.json"))
    parser.add_argument("--max-genes", type=int, default=2048)
    args = parser.parse_args()
    args.output = args.output or args.input_dir / "virtual_cell_training.jsonl"

    matrix_path = next(args.input_dir.glob("*_feature_bc_matrix.h5"))
    contig_path = next(args.input_dir.glob("*_vdj_b_filtered_contig_annotations.csv"))
    receptors = load_receptors(contig_path)
    annotator = MarkerStageAnnotator()
    stage_counts: Counter[str] = Counter()
    paired_counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(matrix_path, "r") as handle, args.output.open("w") as output:
        matrix = handle["matrix"]
        shape = tuple(int(x) for x in matrix["shape"][:])
        x = csc_matrix((matrix["data"][:], matrix["indices"][:], matrix["indptr"][:]), shape=shape)
        names = [v.decode() for v in matrix["features"]["name"][:]]
        types = [v.decode() for v in matrix["features"]["feature_type"][:]]
        barcodes = [v.decode() for v in matrix["barcodes"][:]]

        for col, barcode in enumerate(barcodes):
            start, stop = x.indptr[col], x.indptr[col + 1]
            expression = {
                names[int(row)]: float(value)
                for row, value in zip(x.indices[start:stop], x.data[start:stop])
                if types[int(row)] == "Gene Expression" and value > 0
            }
            stage, confidence = annotator.classify(expression)
            if stage is None:
                continue
            receptor = receptors.get(barcode)
            # A complete conventional light chain is incompatible with a
            # pro/pre-B pseudo-label.  Treat the marker call as ambiguous
            # instead of leaking a mature paired receptor into an early stage.
            if (receptor and receptor.light_aa
                    and stage.value in {"pro-B", "pre-B"}):
                continue
            cell = VirtualBoneMarrowCell(
                cell_id=f"10x_20k_bmmnc:{barcode}",
                stage=stage,
                expression=expression,
                stage_confidence=confidence,
                label_source=annotator.label_source,
                paired_receptor=receptor,
                metadata={"dataset": "10x_20k_bmmnc_5p_ht_v2", "barcode": barcode},
            )
            example = make_training_example(
                cell, stage_regulatory_edges(stage), max_genes=args.max_genes
            )
            output.write(json.dumps(example.to_dict(), separators=(",", ":")) + "\n")
            stage_counts[stage.value] += 1
            if receptor is not None and receptor.heavy_aa and receptor.light_aa:
                paired_counts[stage.value] += 1

    summary = {
        "schema_version": "virtual-bone-marrow-training-1",
        "source": "10x 20k BMMNC 5' HT v2",
        "matrix_sha256": sha256(matrix_path),
        "vdj_sha256": sha256(contig_path),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "label_source": annotator.label_source,
        "labels_are_curated": False,
        "stage_counts": dict(stage_counts),
        "paired_bcr_counts": dict(paired_counts),
        "training_examples": sum(stage_counts.values()),
        "structure_records": 0,
        "affinity_records": 0,
        "structure_status": "not_run: no AlphaFold-compatible backend/artifacts configured",
        "limitations": [
            "Stages are conservative marker-based pseudo-labels, not curated annotations.",
            "Filtered VDJ contigs contain productive receptors and cannot train nonproductive rearrangement.",
            "AlphaFold confidence is not an affinity label and no structure values were fabricated.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

