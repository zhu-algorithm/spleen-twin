"""Acceptance tests for the reconstructed M0 and M1 contracts."""

import json

import numpy as np

from spleen_twin.airr import dumps, receptor_from_airr, receptor_to_airr
from spleen_twin.modules.affinity import SurrogateAffinityOracle
from spleen_twin.modules.receptor import EditingEvent, PairedReceptor
from spleen_twin.modules.vdj import ChainRecombinator
from spleen_twin.organ.spleen import Spleen
from spleen_twin.validation import validate_vdj
from spleen_twin.modules.tolerance import ToleranceOutcome, ToleranceResult


def _paired(seed=17):
    rng = np.random.default_rng(seed)
    h = ChainRecombinator("IGH", rng=rng).rearrange_productive()
    l = ChainRecombinator("IGK", rng=rng).rearrange_productive()
    return PairedReceptor(h, l, [EditingEvent(0, "IGK", l.v_gene, l.j_gene, "initial", True)])


def test_manifest_is_machine_readable_and_replayable(tmp_path):
    spleen = Spleen(seed=42, pair_light_chains=False)
    path = tmp_path / "manifest.json"
    spleen.run_manifest().write(path)
    data = json.loads(path.read_text())
    assert data["seed"] == 42
    assert data["config"]["pair_light_chains"] is False
    assert {p["evidence"] for p in data["parameters"]} == {"model_knob"}


def test_paired_receptor_contains_six_cdrs():
    p = _paired().paratope()
    assert set(p) == {"h_cdr1", "h_cdr2", "h_cdr3", "l_cdr1", "l_cdr2", "l_cdr3"}
    assert all(p.values())


def test_paired_affinity_depends_on_light_chain():
    r1 = _paired(2)
    r2 = _paired(3)
    # Hold the heavy chain fixed; only the light chain differs.
    r2.heavy = r1.heavy
    oracle = SurrogateAffinityOracle()
    epitope = "NLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    assert oracle.raw_score(r1.paratope(), epitope) != oracle.raw_score(r2.paratope(), epitope)


def test_paired_airr_round_trip_is_lossless():
    receptor = _paired()
    record = receptor_to_airr(receptor, "cell-1")
    restored = receptor_from_airr(json.loads(dumps(record)))
    assert restored.to_dict() == receptor.to_dict()


def test_spleen_generates_paired_naive_cells():
    spleen = Spleen(seed=9, pair_light_chains=True)
    stats = spleen.generate_naive_repertoire(n_cells=12)
    assert stats.n_naive == 12
    assert all(c.light_rearrangement and c.light_rearrangement.productive for c in spleen.naive_pool)
    assert all(len(c.paratope()) == 6 for c in spleen.naive_pool)


def test_seed_reproduces_paired_repertoire_exactly():
    a, b = Spleen(seed=123), Spleen(seed=123)
    a.generate_naive_repertoire(8)
    b.generate_naive_repertoire(8)
    sig = lambda s: [(c.rearrangement.seq_nt, c.light_rearrangement.seq_nt) for c in s.naive_pool]
    assert sig(a) == sig(b)


def test_versioned_vdj_validation_passes():
    report = validate_vdj(seed=0, n=1000)
    assert report["schema_version"] == "m0-m1-validation-1"
    assert report["passed"], report


def test_receptor_editing_rearranges_and_records_light_chain():
    class EditOnce:
        def screen(self, paratope, edits_used=0):
            outcome = ToleranceOutcome.EDITED if edits_used == 0 else ToleranceOutcome.NAIVE
            return ToleranceResult(outcome, -8.0, edits_used)

    spleen = Spleen(seed=71, max_light_edit_attempts=2)
    stats = spleen.generate_naive_repertoire(4, tolerance=EditOnce())
    assert stats.n_edited == 4
    assert all(len(c.editing_history) == 2 for c in spleen.naive_pool)
    assert all(c.editing_history[-1].reason == "self_reactivity" for c in spleen.naive_pool)
