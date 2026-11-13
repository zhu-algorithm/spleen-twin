# GeneFormer-X + V(D)J + structure + diffusion maturation stack

## Responsibility boundaries

| Component | Learns | Must not be interpreted as |
|---|---|---|
| GeneFormer-X | B-cell state, regulatory hierarchy, perturbation response | an antibody sequence generator |
| Paired V(D)J Transformer | IGH/IGK/IGL sequence likelihood, junction and heavy-light compatibility | proof of antigen binding |
| Conditional diffusion model | diverse heavy- and light-chain mutation proposals | an affinity measurement |
| AlphaFold-like complex model | plausible complex geometry and confidence | KD, kon or koff |
| Calibrated affinity head | KD/kinetic estimate with uncertainty | experimental validation |
| Virtual germinal centre | repeated SHM, antigen capture, Tfh competition and clonal selection | a guarantee of clinical success |

The implementation contract is in `spleen_twin.modules.ai_maturation`.  A
candidate must retain separate heavy and light sequences throughout the
pipeline.  The structure backend gates clashes and implausible interfaces;
its confidence is never converted directly into affinity.  The affinity head
must be calibrated on measured antibody-antigen data and report uncertainty.

## Training objectives

1. Masked gene/value and regulatory-edge prediction for GeneFormer-X.
2. Autoregressive or masked V(D)J sequence recovery with explicit V/D/J,
   productivity, CDR3 and paired-chain losses.
3. Contrastive matching of true versus shuffled heavy-light pairs.
4. Conditional diffusion denoising over permitted heavy and light mutations.
5. Complex-structure distillation using interface confidence and clash losses.
6. Supervised KD/kon/koff prediction, donor-grouped and antigen-held-out.
7. Multi-round ranking against experimentally observed lineage trajectories.

## Required evaluation

Compare the full stack with expression-only, V(D)J-only, structure-only and
no-diffusion ablations.  Report donor-held-out V/J usage, productive-chain
accuracy, heavy-light retrieval, mutation recovery, antigen-held-out ranking,
Spearman correlation to measured KD, top-k enrichment and uncertainty
calibration.  The decisive test is prospective wet-lab validation.

