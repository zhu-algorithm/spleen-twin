# M0/M1 scientific and software contract

This contract separates simulated mechanisms from measured predictions and is
the acceptance baseline for V(D)J and paired-antibody generation.

## Evidence classes

Every exposed parameter is one of: `measured`, `literature`, `inferred`,
`calibrated`, `surrogate`, or `model_knob`. A run manifest records the seed,
configuration, package/Python versions, parameter records and input hashes.

Affinity values produced by `SurrogateAffinityOracle` are simulated fitness
values. They are not experimental KD predictions. Light-chain CDR1/CDR2
boundaries are inferred from canonical IMGT widths because the packaged
gapped reference covers IGHV only; the `donor_allele` field records this as
`inferred-IMGT-light-boundaries`.

## Reproducibility

- One NumPy generator, rooted at the organ seed, drives recombination and the
  downstream simulation.
- `Spleen.run_manifest()` emits replay metadata.
- Exact sequence equality is expected for the same environment, package/data
  versions, configuration and seed.
- Distributional claims are checked separately by
  `spleen_twin.validation.validate_vdj` and carry explicit intervals.

## Paired receptor contract

- A receptor contains one productive IGH and one productive IGK or IGL chain.
- Heavy and light provenance, junction anatomy and sequences remain separate.
- The affinity paratope exposes six named loops (`h_cdr1..3`, `l_cdr1..3`).
- Somatic hypermutation currently changes the heavy sequence; the paired light
  chain and its editing history remain attached to every descendant.
- Light-chain editing creates a new productive IGK/IGL rearrangement, logs the
  attempt and reason, then re-runs tolerance. Attempt limits are explicit.

## AIRR-aligned exchange

`spleen_twin.airr` stores each chain using AIRR field names and preserves
project-specific junction events, flags and editing history in namespaced
extensions. Its round trip is lossless for all fields used by this simulator.

## Versioned invariants

For the reference seeded IGH sample:

- in-frame fraction: 0.28–0.40;
- productive fraction: 0.12–0.32;
- mean CDR-H3 length excluding anchors: 12–19 amino acids.

These are plausibility/regression intervals, not universal human population
confidence intervals. Changing them requires a cited validation change.
