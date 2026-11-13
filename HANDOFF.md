# Spleen Twin algorithm handoff — v0.2.0

This source package is intended for continued research and engineering. It
contains no credentials, no proprietary AlphaGenome or AlphaFold weights, and
no multi-gigabyte external matrices.

## Implemented

- human IGH, IGK and IGL rearrangement with OLGA event distributions;
- productive/nonproductive checkpoints, paired heavy/light receptors and
  light-chain receptor editing;
- antigen presentation, T-cell help, germinal-centre SHM/selection, class
  switching, plasma/memory output and developability gates;
- SHM of both heavy and light chains;
- HSC → CLP → pro-B → pre-B → immature B → mature B virtual-cell contracts;
- Geneformer V1 frozen-encoder pilot and a regulatory-aware stage head;
- GeneFormer-X/paired-VDJ/diffusion/structure/affinity integration contracts;
- AlphaGenome 2 stage-gated regulatory priors that reweight OLGA segment
  selection without replacing junction mechanics;
- reproducible spleen and bone-marrow data download/validation scripts.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

The handoff release was verified with 43 passing tests.

Run a small campaign:

```bash
.venv/bin/python examples/run_campaign.py
```

## Rebuild external data and the Geneformer pilot

```bash
scripts/download_bone_marrow_vdj.sh
scripts/download_spleen_multiomics.sh
scripts/download_geneformer_v1.sh
PYTHONPATH=src .venv/bin/python scripts/build_bone_marrow_training_set.py
PYTHONPATH=src .venv/bin/python scripts/train_geneformer_stage_pilot.py
```

CELLxGENE Census extraction is handled by `scripts/fetch_spleen_census.py`.
Large outputs belong under `data/external/` and are intentionally excluded.

## External backends

- **AlphaGenome 2:** requires `ALPHA_GENOME_API_KEY`. Configure it in the
  environment, never in source control. The public non-commercial API terms
  prohibit using direct outputs to train another ML model; the current code
  therefore uses outputs only as frozen inference-time priors.
- **AlphaFold-like structure backend:** the repository defines typed adapters
  but does not bundle a proprietary model or fabricate structure confidence.
- **Geneformer:** `scripts/download_geneformer_v1.sh` downloads the official
  Apache-2.0 V1 10M checkpoint and dictionaries. The pilot metrics are
  resubstitution diagnostics, not validated performance.

## Scientific limits

The local 10x marrow data include 1,489 productive paired BCR cells but very
few early cells (1 CLP, 2 pro-B and 3 pre-B under conservative gates). They do
not contain enough early-lineage, nonproductive or genomic VDJ observations to
prove that AlphaGenome 2 improves rearrangement accuracy. A definitive study
needs multi-donor stage-enriched scRNA+scATAC, personalized IG haplotypes and
unfiltered/genomic VDJ, followed by donor/study/clonotype-held-out ablations.

## Key entry points

- `src/spleen_twin/modules/vdj.py` — mechanistic rearrangement
- `src/spleen_twin/modules/alphagenome_vdj.py` — regulatory prior contract
- `src/spleen_twin/modules/ai_maturation.py` — multimodal maturation contract
- `src/spleen_twin/organ/bone_marrow.py` — six-stage virtual cells
- `src/spleen_twin/organ/spleen.py` — organ-level campaign
- `docs/` — model equations, milestones and architecture notes
- `data/*.json` — compact provenance and current-status manifests

## Licensing

Repository source is MIT licensed. External datasets, pretrained checkpoints
and hosted model outputs retain their own licences and terms. Review those
terms before redistribution or commercial use.
