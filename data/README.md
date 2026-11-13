# Human spleen atlas staging area

This directory holds the compact, versioned metadata for the virtual-spleen
atlas. Large downloaded matrices live under `data/external/` and are ignored
by Git; they can be reproduced with the scripts in `scripts/`.

## Verified local sources

| Source | Modality | Verified matrix | Purpose |
|---|---|---:|---|
| CELLxGENE Census 2025-11-08 | sc/snRNA-seq | 577,677 cells; 58 donors | cell composition and expression grounding |
| HuBMAP spleen RNA 2025-05-19 | sc/snRNA-seq | 108,929 × 60,286 | full transcriptome reference and donor covariates |
| HuBMAP spleen CODEX 2024-11-07 | spatial proteomics | 3,939,679 × 34 | spatial neighbourhood and white/red-pulp architecture |
| 10x human bone marrow 5' HT v2 | GEX + ADT + BCR V(D)J | 17,645 cells; 36,601 genes; 1,489 productive heavy-light pairs | V(D)J-linked B-cell state, clonotype and productive heavy-light pairing |

Checksums for downloaded HuBMAP matrices are written to
`data/external/hubmap/SHA256SUMS`. Machine-readable provenance, observed
matrix dimensions, source URLs and integration rules are in
`spleen_atlas_sources.json`.

The bone-marrow V(D)J core can be reproduced with
`scripts/download_bone_marrow_vdj.sh`. Its checksums and observed matrix,
chain and heavy-light pairing statistics are written beside the downloaded
files. Source selection and expansion accessions are recorded in
`bone_marrow_atlas_sources.json`.

The AlphaGenome 2 V(D)J integration state, usage boundary and stage-specific
conditioning readiness are recorded in `alphagenome2_vdj_status.json`.

## Admission rules

Do not add headline cell counts across sources until dataset, donor and cell
barcode overlap has been checked. Keep raw and normalised values in separate
layers. Every integrated observation must retain source, dataset, donor,
assay, tissue, disease and licence fields, and every mapped cell type, gene or
protein marker must retain its original label.

GEO GSE299043 is intentionally excluded from the spleen-only core for now.
Its public libraries pool spleen with several other organs and the inspected
H5AD contains no cell-level tissue metadata, so including it would introduce
unresolvable non-spleen cells.

