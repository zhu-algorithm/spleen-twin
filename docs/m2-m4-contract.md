# M2–M4 implementation and validation contract

## M2 — Virtual Spleen

The organ exposes named anatomical compartments, a documented event clock and
validated non-negative state variables. Multi-GC runs use disjoint cell and
clone namespaces and report seed/GC distributions as simulation uncertainty,
not experimental confidence intervals. Parameter sweeps reuse identical seeds
across conditions.

Atlas expression is optional and never silently imputed. The packaged atlas
contains composition; `scripts/fetch_spleen_census.py` now checkpoints every
expression block and resumes failed/incomplete extractions. Expression-backed
features must check `AtlasReference.expression_available` and provenance.

## M3 — maturation and candidates

Paired heavy/light CDRs contribute to selection. The light chain and editing
history are inherited through every GC descendant. Multi-GC lineage IDs cannot
collide. Affinity results state whether they are `surrogate`, `structural`, or
`experimental`; backend failure raises and cannot become a fabricated score.

Candidate cards contain both chains, lineage/isotype, affinity evidence and
uncertainty, developability tier, failures, warnings and editing history.
Developability remains explicitly heuristic.

## M4 — closed loop and release

Campaign rounds have an atomic, versioned registry. Lineage leakage is checked
before training. Evaluation records whether model and baseline budgets match
and stores the preregistered hit threshold. Active-learning selections retain
their affinity, uncertainty and diversity rationale. Failures remain data.

Release acceptance requires:

- complete test suite passes;
- benchmark is replayable from a clean installation;
- M0/M1 and M2–M4 contracts are included;
- surrogate and inferred outputs remain labelled;
- no external atlas expression is claimed unless extraction provenance exists.
