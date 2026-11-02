"""Lossless AIRR-aligned serialization for paired simulated receptors."""

from __future__ import annotations

import json
from dataclasses import asdict

from .modules.receptor import EditingEvent, PairedReceptor
from .modules.vdj import Junction, Rearrangement


def _chain_record(r: Rearrangement) -> dict:
    return {
        "sequence": r.seq_nt, "sequence_aa": r.seq_aa,
        "v_call": r.v_gene, "d_call": r.d_gene, "j_call": r.j_gene,
        "junction": r.cdr3_nt, "junction_aa": r.cdr3_aa,
        "productive": r.productive, "locus": r.locus,
        "spleen_twin_junction": asdict(r.junction),
        "spleen_twin_flags": {
            "in_frame": r.in_frame, "has_stop": r.has_stop,
            "conserved_c": r.conserved_c, "conserved_w_f": r.conserved_w_f,
            "v_prefix_nt": r.v_prefix_nt, "fr4_nt": r.fr4_nt,
        },
    }


def receptor_to_airr(receptor: PairedReceptor, sequence_id: str) -> dict:
    return {
        "sequence_id": sequence_id,
        "receptor_type": "paired_igh_igkl",
        "heavy": _chain_record(receptor.heavy),
        "light": _chain_record(receptor.light),
        "spleen_twin_editing_history": [asdict(e) for e in receptor.editing_history],
        "schema_version": "spleen-twin-paired-airr-1",
    }


def _chain_from_record(d: dict) -> Rearrangement:
    flags = d["spleen_twin_flags"]
    return Rearrangement(
        locus=d["locus"], v_gene=d["v_call"], d_gene=d.get("d_call"),
        j_gene=d["j_call"], junction=Junction(**d["spleen_twin_junction"]),
        cdr3_nt=d["junction"], cdr3_aa=d["junction_aa"], seq_nt=d["sequence"],
        seq_aa=d["sequence_aa"], productive=bool(d["productive"]), **flags,
    )


def receptor_from_airr(record: dict) -> PairedReceptor:
    if record.get("schema_version") != "spleen-twin-paired-airr-1":
        raise ValueError("unsupported paired AIRR schema")
    return PairedReceptor(
        _chain_from_record(record["heavy"]), _chain_from_record(record["light"]),
        [EditingEvent(**e) for e in record.get("spleen_twin_editing_history", [])],
    )


def dumps(record: dict) -> str:
    return json.dumps(record, sort_keys=True)


__all__ = ["receptor_to_airr", "receptor_from_airr", "dumps"]
