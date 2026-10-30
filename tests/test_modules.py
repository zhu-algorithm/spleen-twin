"""Mechanism tests: each asserts a published property of the biology."""

import numpy as np
import pytest

from spleen_twin.modules import affinity, antigen, regions, shm, vdj
from spleen_twin.modules.germinal_center import GerminalCenter


@pytest.fixture(scope="module")
def igh():
    return vdj.ChainRecombinator("IGH", rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def sample(igh):
    return [r for r in (igh.rearrange() for _ in range(4000)) if r is not None]


def test_germline_counts(igh):
    assert len(igh.v_genes) > 50
    assert len(igh.d_genes) > 20
    assert len(igh.j_genes) >= 6


def test_one_third_in_frame(sample):
    """Junctional diversity is frame-agnostic, so ~1/3 of joins are in frame."""
    frac = sum(r.in_frame for r in sample) / len(sample)
    assert 0.28 < frac < 0.40, frac


def test_productive_fraction_is_plausible(sample):
    frac = sum(r.productive for r in sample) / len(sample)
    assert 0.12 < frac < 0.32, frac


def test_cdr3_length_distribution(sample):
    """Human CDR-H3 averages ~15-16 aa excluding the two IMGT anchors."""
    lens = [r.cdr3_len_aa for r in sample if r.productive]
    mean_excl_anchors = float(np.mean(lens)) - 2
    assert 12.0 < mean_excl_anchors < 19.0, mean_excl_anchors


def test_translation_has_framework_motifs(sample):
    """A correctly assembled VH ends in the FR4 WGxG motif."""
    prod = [r for r in sample if r.productive]
    ok = sum(1 for r in prod if "WG" in r.seq_aa[-14:] and r.seq_aa[-1] in "SA")
    assert ok / len(prod) > 0.8


def test_p_nucleotides_are_palindromic(igh):
    for i in range(len(igh.genomic.genV)):
        germ = igh.genomic.genV[i][1]
        ext = igh.genomic.cutV_genomic_CDR3_segs[i][len(germ):]
        if len(ext) == vdj.P_LEN and len(germ) >= vdj.P_LEN:
            assert ext == vdj.revcomp(germ[-vdj.P_LEN:])


def test_imgt_regions_are_contiguous_and_complete(sample):
    ann = regions.ImgtAnnotator()
    prod = [r for r in sample if r.productive][:200]
    maps = [ann.annotate(r) for r in prod]
    assert sum(m.annotated for m in maps) / len(maps) > 0.9
    for m, r in zip(maps, prod):
        assert len(m.labels) == len(r.seq_nt)
    counts = maps[0].counts()
    for reg in ("fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"):
        assert counts.get(reg, 0) > 0


def test_s5f_prefers_classic_aid_hotspots(sample):
    """WRCY/RGYW motifs must be more mutable than the rest of the sequence."""
    import re
    model = shm.S5FModel()
    assert len(model.mutability) == 1024
    seq = next(r.seq_nt for r in sample if r.productive)
    prof = model.mutability_profile(seq)
    hot = set()
    for m in re.finditer(r"(?=([AT][AG]C[CT]))", seq):
        hot.add(m.start() + 2)
    for m in re.finditer(r"(?=([AG]G[CT][AT]))", seq):
        hot.add(m.start() + 1)
    cold = set(range(len(seq))) - hot
    assert np.mean([prof[i] for i in hot]) > 2.0 * np.mean([prof[i] for i in cold])


def test_shm_rate_matches_expectation(sample):
    eng = shm.HypermutationEngine(rng=np.random.default_rng(1))
    seq = next(r.seq_nt for r in sample if r.productive)
    counts = [len(eng.mutate(seq, divisions=10)[1]) for _ in range(400)]
    expected = shm.DEFAULT_SHM_RATE * len(seq) * 10
    assert 0.7 * expected < np.mean(counts) < 1.3 * expected


def test_affinity_calibration_spans_reported_naive_range(sample):
    ag = antigen.hen_egg_lysozyme()
    ep = ag.immunodominant()
    ann = regions.ImgtAnnotator()
    oracle = affinity.SurrogateAffinityOracle()
    paratopes = []
    for r in sample:
        if not r.productive:
            continue
        m = ann.annotate(r)
        if m.annotated:
            paratopes.append(affinity.paratope_from(m, r.seq_aa))
        if len(paratopes) >= 400:
            break
    raws = [oracle.raw_score(p, ep.sequence)[0] for p in paratopes]
    oracle.calibrate(raws)
    kds = np.array([oracle.score(p, ep.sequence).log10_kd for p in paratopes])
    assert -6.8 < np.median(kds) < -5.8
    assert kds.min() > -9.0, "naive repertoire should not already contain sub-nM binders"


def test_antigen_is_mature_lysozyme():
    ag = antigen.hen_egg_lysozyme()
    assert len(ag.sequence) == 129
    assert ag.sequence.startswith("KVFGRCELAA")
    assert all(e.sequence == ag.sequence[e.start - 1:e.end] for e in ag.epitopes)


def test_cdr_index_matches_region_labels(sample):
    ann = regions.ImgtAnnotator()
    r = next(x for x in sample if x.productive and ann.annotate(x).annotated)
    m = ann.annotate(r)
    idx = GerminalCenter.cdr_index_from_regions(m.labels)
    for loop, positions in idx.items():
        for p in positions:
            assert m.labels[p * 3] == loop
