"""Tests for the closed loop's data-quality guarantees."""

import pytest

from spleen_twin.loop.curate import AntibodyRecord, Curator, REQUIRED_METADATA


def _record(sid="a", seq="ACGT", kd=1e-9, meta=None):
    return AntibodyRecord(
        sequence_id=sid, sequence=seq, sequence_aa="QVQL",
        v_call="IGHV1-18*01", d_call=None, j_call="IGHJ4*02",
        junction_aa="CARDYW", productive=True, cdr3_aa="CARDYW",
        kd_molar=kd, log10_kd=-9.0, binder=True,
        metadata=meta if meta is not None else {f: "x" for f in REQUIRED_METADATA} | {"assay": "BLI"},
    )


def test_incomplete_metadata_is_rejected():
    c = Curator()
    assert not c.add(_record(meta={"campaign_id": "c1"}))
    assert c.rejected[0]["problems"]


def test_unknown_assay_is_rejected():
    meta = {f: "x" for f in REQUIRED_METADATA} | {"assay": "vibes"}
    c = Curator()
    assert not c.add(_record(meta=meta))


def test_duplicate_sequences_are_dropped():
    c = Curator()
    assert c.add(_record(sid="a", seq="ACGTACGT"))
    assert not c.add(_record(sid="b", seq="ACGTACGT"))


def test_negatives_are_kept_by_default():
    c = Curator()
    r = _record(sid="neg")
    r.binder = False
    assert c.add(r)
    assert c.summary()["n_non_binders"] == 1


def test_negatives_can_be_dropped_only_explicitly():
    c = Curator(keep_negatives=False)
    r = _record(sid="neg")
    r.binder = False
    assert not c.add(r)


def test_mixing_assay_families_is_flagged():
    c = Curator()
    m1 = {f: "x" for f in REQUIRED_METADATA} | {"assay": "BLI"}
    m2 = {f: "x" for f in REQUIRED_METADATA} | {"assay": "ELISA"}
    c.add(_record(sid="a", seq="AAAA", meta=m1))
    c.add(_record(sid="b", seq="CCCC", meta=m2))
    assert c.summary()["comparable"] is False


def test_lineage_split_separates_lineages():
    from spleen_twin.loop.train import lineage_key, lineage_split

    recs = []
    for fam in range(6):
        for i in range(10):
            r = _record(sid=f"{fam}-{i}", seq=f"ACGT{fam}{i}")
            r.v_call = f"IGHV{fam}-1*01"
            r.junction_aa = f"CAR{fam}DYW"
            recs.append(r)
    tr, te, n_lin = lineage_split(recs, test_fraction=0.34, seed=0)
    assert n_lin == 6
    train_keys = {lineage_key(recs[i]) for i in tr}
    test_keys = {lineage_key(recs[i]) for i in te}
    assert not (train_keys & test_keys)


def test_protein_only_records_are_not_all_duplicates():
    """Designed variants carry no nucleotide sequence; they must still be distinct."""
    c = Curator()
    added = 0
    for i, cdr3 in enumerate(["CARDYW", "CARDFW", "CARDHW", "CARDKW"]):
        r = _record(sid=f"d{i}", seq="")
        r.sequence = ""
        r.sequence_aa = f"QVQL{cdr3}WGQG"
        r.cdr3_aa = cdr3
        added += int(c.add(r))
    assert added == 4, c.rejected
