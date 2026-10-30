"""Run one immunisation campaign through the virtual spleen.

    python examples/run_campaign.py --rounds 30 --seed 20261030
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from spleen_twin.modules import antigen as antigen_mod
from spleen_twin.modules import tolerance as tol
from spleen_twin.modules.developability import DevelopabilityPanel
from spleen_twin.modules.reformatting import ReformattingModule
from spleen_twin.organ import Spleen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--naive", type=int, default=2500)
    ap.add_argument("--founders", type=int, default=150)
    ap.add_argument("--epitope", default=None)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ag = antigen_mod.hen_egg_lysozyme(dose_nm=10.0)
    epitope = (next(e for e in ag.epitopes if e.name == args.epitope)
               if args.epitope else ag.immunodominant())
    print(f"antigen {ag.name} ({len(ag.sequence)} aa), epitope {epitope.name}: {epitope.sequence}")

    spleen = Spleen(seed=args.seed,
                    gc_kwargs=dict(carrying_capacity=1500, max_light_zone_cells=7000))

    # Bootstrap the affinity scale and the tolerance thresholds on an
    # unscreened sample of this spleen's own repertoire.
    boot = Spleen(seed=args.seed + 1)
    boot.generate_naive_repertoire(n_cells=800)
    spleen.oracle.calibrate([spleen.oracle.raw_score(c.paratope(), epitope.sequence)[0]
                             for c in boot.naive_pool])
    spleen._calibrated = True

    checkpoint = tol.ToleranceCheckpoint(spleen.oracle, tol.load_self_panel(), rng=spleen.rng)
    cal = checkpoint.calibrate([c.paratope() for c in boot.naive_pool[:500]])
    print(f"tolerance thresholds: delete {cal['delete_threshold_log10_kd']:.2f}, "
          f"anergy {cal['anergy_threshold_log10_kd']:.2f} (log10 KD)")

    rep = spleen.generate_naive_repertoire(n_cells=args.naive, tolerance=checkpoint)
    screened = rep.n_deleted + rep.n_anergic + rep.n_naive
    print(f"repertoire: {rep.productive_fraction:.1%} productive | "
          f"deleted {rep.n_deleted/screened:.1%}, anergic {rep.n_anergic/screened:.1%}, "
          f"naive {rep.n_naive/screened:.1%}")

    result = spleen.immunise(ag, epitope_name=epitope.name,
                             n_rounds=args.rounds, max_founders=args.founders)
    h = result.gc_history
    print(f"GC: {result.n_founders} founders -> {h[-1].n_clones} clones | "
          f"median KD {10 ** h[-1].median_log10_kd:.2e} M, best {10 ** h[-1].best_log10_kd:.2e} M")

    # Harvest, gate and reformat the lead panel.
    panel = DevelopabilityPanel(apc_module=spleen.apc)
    # Anchor the manufacturability scales on unmutated germline paratopes.
    pcal = panel.calibrate([c.paratope() for c in boot.naive_pool[:800]])
    print(f"developability baseline: patch ref {pcal['patch_reference']:.3f} "
          f"+/- {pcal['patch_spread']:.3f} (from {pcal['n_sampled']} germline paratopes)")
    reformatter = ReformattingModule(constant_region="IgG1", platform="VelocImmune")
    harvested = spleen.harvest_single_b_cells(result)
    harvested.sort(key=lambda c: c.log10_kd)
    # Clonal siblings differ by a few mutations and many are identical at
    # the protein level; a lead panel lists unique molecules.
    seen, unique = set(), []
    for c in harvested:
        if c.seq_aa not in seen:
            seen.add(c.seq_aa)
            unique.append(c)
    print(f"harvested {len(harvested)} paired clones -> {len(unique)} unique VH sequences")

    leads = []
    for i, c in enumerate(unique[:20]):
        construct = reformatter.reformat(c, name=f"ST-{i:03d}")
        report = panel.evaluate(construct.vh_aa, c.paratope(), 10 ** c.log10_kd)
        leads.append({
            "name": construct.name, "v_gene": construct.germline_v,
            "cdr3_aa": c.rearrangement.cdr3_aa, "isotype": c.isotype,
            "kd_molar": 10 ** c.log10_kd, "n_mutations": c.n_mutations,
            "humanness": round(construct.humanness, 4),
            "constant_swap_performed": construct.constant_swap_performed,
            "tier": report.tier.value, "tm_celsius": round(report.tm_celsius, 1),
            "titre_g_l": round(report.titre_g_l, 2),
            "failures": report.failures, "warnings": report.warnings,
            "vh_aa": construct.vh_aa,
        })

    tiers = {}
    for l in leads:
        tiers[l["tier"]] = tiers.get(l["tier"], 0) + 1
    print(f"lead panel {len(leads)} | developability tiers {tiers}")

    # Mean nucleotides per IMGT region, needed to turn raw mutation counts
    # into per-site rates (CDRs are far shorter than the frameworks).
    import collections as _c
    region_nt = _c.Counter()
    n_counted = 0
    for c in unique[:200]:
        for lab in c.regions:
            region_nt[lab] += 1
        n_counted += 1
    region_nt_mean = {k: v / max(1, n_counted) for k, v in region_nt.items()}
    cdr_nt = sum(v for k, v in region_nt_mean.items() if k.startswith("cdr"))
    fr_nt = sum(v for k, v in region_nt_mean.items() if k.startswith("fwr"))
    print(f"mean region sizes: CDR {cdr_nt:.0f} nt, framework {fr_nt:.0f} nt")

    with open(os.path.join(args.outdir, "campaign_summary.json"), "w") as fh:
        json.dump({
            "seed": args.seed, "antigen": ag.name, "epitope": epitope.name,
            "repertoire": rep.__dict__,
            "region_nt_mean": region_nt_mean,
            "cdr_nt_mean": cdr_nt, "fr_nt_mean": fr_nt,
            "gc_rounds": [s.__dict__ for s in h],
            "serum_titre": result.serum_titre,
            "leads": leads,
        }, fh, indent=1)
    with open(os.path.join(args.outdir, "lead_panel.fasta"), "w") as fh:
        for l in leads:
            fh.write(f">{l['name']} V={l['v_gene']} KD={l['kd_molar']:.2e} "
                     f"tier={l['tier']} isotype={l['isotype']}\n{l['vh_aa']}\n")
    print(f"wrote {args.outdir}/campaign_summary.json and lead_panel.fasta")


if __name__ == "__main__":
    main()
