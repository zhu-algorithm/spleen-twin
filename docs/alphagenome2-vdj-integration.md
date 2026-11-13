# AlphaGenome 2 integration for V(D)J accuracy

AlphaGenome 2 is used as a frozen, cell-state-conditioned regulatory prior.
It does not replace the V(D)J generator and does not predict affinity.

```text
raw UMI counts from one marrow stage (up to 256 cells)
                         + personalized/reference IG locus DNA
                                      |
                              AlphaGenome 2
                                      |
           accessibility / enhancer / methylation / 3D-contact evidence
                                      |
                  calibrated per-segment regulatory weights
                                      |
GeneFormer-X cell state ----------------+---------------- RSS/12-23 evidence
                                      |
                         OLGA + VDJ Transformer
                                      |
          segment choice -> deletion -> P/N insertion -> productivity
                                      |
                    paired heavy/light receptor
                                      |
                 structure, antigen and GC maturation
```

## Stage gates

| Stage | AlphaGenome 2 role | V(D)J action |
|---|---|---|
| HSC | reference IG-locus baseline | none |
| CLP | locus priming/accessibility | no complete receptor target |
| pro-B | IGH accessibility, enhancer and recombination-centre contact | D-J then V-DJ segment choice |
| pre-B | IGK/IGL accessibility after a productive heavy chain | light-chain V-J choice |
| immature B | light-locus reopening during receptor editing | optional IGK/IGL editing only |
| mature B | RAG-off negative control | no new V(D)J |

The implementation modifies only OLGA's V, joint D/J or joint V/J sampling
probabilities. Deletion distributions, palindromic nucleotides, TdT insertions,
reading frame, stop codons and conserved-anchor checks remain mechanistic.

## What is required before claiming improvement

1. Stage-enriched, multi-donor pro-B and pre-B raw UMI counts.
2. scATAC/CUT&Tag or related measured tracks to calibrate AlphaGenome outputs.
3. Unfiltered or genomic V(D)J containing incomplete and nonproductive events.
4. Personalized diploid IG haplotypes and versioned IMGT/Ensembl coordinates.
5. Donor-, study- and clonotype-held-out comparison against the unchanged OLGA
   baseline using event log loss, V/D/J usage divergence, calibration and
   productive-fraction accuracy.

The local 10x dataset contains only one confidently labelled CLP, two pro-B
and three pre-B cells, and its filtered VDJ contigs are productive. It can
exercise the interface but cannot validate an accuracy gain.

## Usage boundary

The public AlphaGenome 2 API is research-use and requires an API key. Its
non-commercial terms state that direct outputs must not be used to train other
machine-learning models. The code therefore uses public-API outputs only for
inference-time reranking and raises an error if they are requested as training
features. A separately documented commercial licence may enable another path.

