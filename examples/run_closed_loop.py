"""Run the wet-dry closed loop on top of the virtual spleen.

    python examples/run_closed_loop.py --loops 5

Each loop: generate -> curate -> train -> design -> test -> evaluate.
Both a model-guided arm and a random baseline arm are proposed and
measured every round, so the contribution of the model is always visible.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from spleen_twin.loop import curate as curate_mod
from spleen_twin.loop import design as design_mod
from spleen_twin.loop import evaluate as eval_mod
from spleen_twin.loop import generate as gen_mod
from spleen_twin.loop import test as test_mod
from spleen_twin.loop import train as train_mod
from spleen_twin.modules import antigen as antigen_mod
from spleen_twin.modules import tolerance as tol
from spleen_twin.modules.developability import DevelopabilityPanel
from spleen_twin.organ import Spleen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--outdir", default="results/loop")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ag = antigen_mod.hen_egg_lysozyme()
    epitope = ag.immunodominant()

    spleen = Spleen(seed=args.seed,
                    gc_kwargs=dict(carrying_capacity=1200, max_light_zone_cells=6000))
    boot = Spleen(seed=args.seed + 1)
    boot.generate_naive_repertoire(n_cells=800)
    spleen.oracle.calibrate([spleen.oracle.raw_score(c.paratope(), epitope.sequence)[0]
                             for c in boot.naive_pool])
    spleen._calibrated = True
    checkpoint = tol.ToleranceCheckpoint(spleen.oracle, tol.load_self_panel(), rng=spleen.rng)
    checkpoint.calibrate([c.paratope() for c in boot.naive_pool[:500]])
    spleen.generate_naive_repertoire(n_cells=2000, tolerance=checkpoint)

    # --- 1. GENERATE: one in-vivo campaign seeds the loop ---
    result = spleen.immunise(ag, n_rounds=args.rounds, max_founders=120)
    panel = DevelopabilityPanel(apc_module=spleen.apc)
    panel.calibrate([c.paratope() for c in boot.naive_pool[:800]])
    records = gen_mod.records_from_campaign("campaign-001", spleen, ag, result,
                                            epitope.name, panel=panel)

    # --- 2. CURATE ---
    curator = curate_mod.Curator(keep_negatives=True)
    for r in records:
        curator.add(r)
    print(f"curated {curator.summary()}")

    history = []
    for loop in range(1, args.loops + 1):
        data = curator.records
        # --- 3. TRAIN ---
        model = train_mod.train_affinity_model(data, seed=args.seed + loop)
        # --- 4. DESIGN: model arm and baseline arm on the same parents ---
        parents = sorted([r for r in data if r.log10_kd is not None],
                         key=lambda r: r.log10_kd)[:12]
        parent_best = min(r.log10_kd for r in parents)
        prop_model = design_mod.propose_model_guided(parents, model, seed=args.seed + loop)
        prop_base = design_mod.propose_random(parents, seed=args.seed + 1000 + loop)
        # --- 5. TEST ---
        meta = gen_mod.campaign_metadata(f"loop-{loop:02d}", spleen, ag, epitope.name)
        measured = test_mod.measure_proposals(prop_model + prop_base, spleen.oracle,
                                              epitope.sequence, meta, loop, panel=panel)
        # --- 6. EVALUATE, then feed everything back including failures ---
        ev = eval_mod.evaluate_round(loop, measured, parent_best)
        ev.n_failures_logged = eval_mod.log_failures(
            measured, parent_best, os.path.join(args.outdir, f"failures_round{loop}.json"))
        for r in measured:
            curator.add(r)

        history.append(ev.__dict__)
        print(f"loop {loop}: model best {ev.best_model_log10_kd:.2f} vs "
              f"baseline {ev.best_baseline_log10_kd:.2f} "
              f"({'model wins' if ev.model_beats_baseline else 'BASELINE WINS'}) | "
              f"test rho {model.metrics['test_spearman']:.2f} | "
              f"failures {ev.n_failures_logged}")

    curator.to_airr_tsv(os.path.join(args.outdir, "curated_records.tsv"))
    curator.to_jsonl(os.path.join(args.outdir, "curated_records.jsonl"))
    curator.write_rejections(os.path.join(args.outdir, "rejected_records.json"))
    with open(os.path.join(args.outdir, "loop_history.json"), "w") as fh:
        json.dump({"history": history, "curation": curator.summary()}, fh, indent=1)
    print(f"wrote {args.outdir}/curated_records.tsv and loop_history.json")


if __name__ == "__main__":
    main()
