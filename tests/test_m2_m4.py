"""Acceptance tests for Virtual Spleen, candidate output and closed-loop release."""

import json

import pytest

from spleen_twin.candidates import card_from_cell, lineage_edges
from spleen_twin.loop.design import Proposal, select_active_learning
from spleen_twin.loop.registry import CampaignRegistry, CampaignRound
from spleen_twin.modules.affinity import CallbackAffinityOracle
from spleen_twin.modules.antigen import hen_egg_lysozyme
from spleen_twin.modules.developability import DevelopabilityPanel
from spleen_twin.organ.atlas import load_reference
from spleen_twin.organ.dynamics import OrganState, TRANSITIONS, aggregate_replicates
from spleen_twin.organ.spleen import Spleen
from spleen_twin.sensitivity import monotonic_effect, parameter_sweep


def test_atlas_missing_expression_is_explicit():
    atlas = load_reference()
    assert atlas.validate() == []
    assert atlas.expression_available is False
    assert atlas.expression("naive B cell", "MS4A1") is None
    assert atlas.summary()["expression_available"] is False


def test_compartment_contract_and_uncertainty():
    assert len(TRANSITIONS) >= 8
    state = OrganState(time_hours=8, antigen_nm=1, tfh_help=0.4, naive_b=10)
    assert state.to_dict()["time_hours"] == 8
    with pytest.raises(ValueError):
        OrganState(antigen_nm=-1).validate()
    summary = aggregate_replicates([1, 2, 3])
    assert summary["n"] == 3 and summary["q50"] == 2


def _short_campaign(seed=41, n_gcs=2):
    spleen = Spleen(seed=seed, n_germinal_centers=n_gcs,
                    gc_kwargs={"carrying_capacity": 30, "max_light_zone_cells": 90})
    spleen.generate_naive_repertoire(100)
    antigen = hen_egg_lysozyme()
    result = spleen.immunise(antigen, n_rounds=2, max_founders=20)
    return spleen, antigen, result


def test_multi_gc_ids_uncertainty_and_organ_state():
    spleen, _, result = _short_campaign()
    ids = [c.cell_id for c in result.plasma_cells + result.memory_cells]
    assert len(ids) == len(set(ids))
    assert result.uncertainty["n_germinal_centers"] == 2
    assert result.uncertainty["final_best_log10_kd"]["n"] == 2
    assert result.organ_state.time_hours == 16
    assert result.organ_state.plasma_b == len(result.plasma_cells)


def test_candidate_card_is_paired_and_evidence_typed():
    spleen, antigen, result = _short_campaign(seed=42, n_gcs=1)
    cell = (result.plasma_cells + result.memory_cells)[0]
    card = card_from_cell(cell, spleen.oracle, antigen.immunodominant().sequence,
                          DevelopabilityPanel(apc_module=spleen.apc))
    assert card.heavy_sequence_aa and card.light_sequence_aa
    assert card.affinity_source == "surrogate"
    assert card.schema_version == "candidate-card-1"
    assert lineage_edges([cell])[0]["cell_id"] == cell.cell_id


def test_external_affinity_backend_never_fabricates_failure():
    oracle = CallbackAffinityOracle(lambda p, e: (1e-9, 0.2), "experimental", "LIMS:test")
    result = oracle.score({"cdr3": "CARDYW"}, "PEPTIDE")
    assert result.source_type == "experimental" and result.uncertainty_log10 == 0.2
    bad = CallbackAffinityOracle(lambda p, e: 0, "structural", "bad")
    with pytest.raises(ValueError):
        bad.score({"cdr3": "CARDYW"}, "PEPTIDE")


def test_sensitivity_grid_is_replayable():
    points = parameter_sweep(lambda p, seed: p["dose"] + seed,
                             {"dose": [1.0, 2.0, 3.0]}, seeds=(0, 1))
    assert [p.summary["mean"] for p in points] == [1.5, 2.5, 3.5]
    assert monotonic_effect(points, "dose")["direction"] == "increasing"


def test_active_learning_is_diverse_and_auditable():
    ps = [Proposal("p", s, "A", "B", s, "V", s, 1, "model_guided",
                   predicted_log10_kd=-8 - i * 0.1, prediction_uncertainty=i * 0.2)
          for i, s in enumerate(["CARAAA", "CARAAT", "CRRRRR", "CWWWWW"])]
    chosen = select_active_learning(ps, 2)
    assert len(chosen) == 2
    assert all(p.acquisition_score is not None and p.selection_reason for p in chosen)


def test_campaign_registry_round_trip(tmp_path):
    registry = CampaignRegistry("campaign-1")
    registry.add(CampaignRound(1, "abc", 20, 10, 10,
                               {"model_beats_baseline": False}, [{"reason": "regression"}]))
    path = tmp_path / "registry.json"
    registry.write(path)
    restored = CampaignRegistry.read(path)
    assert restored.to_dict()["rounds"][0]["failures"]
    with pytest.raises(ValueError):
        restored.add(CampaignRound(1, "def", 1, 1, 1, {}))
