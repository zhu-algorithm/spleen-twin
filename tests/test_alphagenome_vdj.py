import numpy as np
import pytest

from spleen_twin.modules.alphagenome_vdj import (
    AlphaGenome2Status, AlphaGenome2VDJPrior, GenomicInterval,
    SegmentRegulatoryEvidence, UnavailableAlphaGenome2Backend,
)
from spleen_twin.modules.vdj import ChainRecombinator


def _evidence(segment="IGHV1-2*01", locus="IGH", score=0.8):
    return SegmentRegulatoryEvidence(
        segment=segment,
        locus=locus,
        interval=GenomicInterval("chr14", 1_000, 1_300),
        accessibility=score,
        enhancer_activity=score,
        recombination_center_contact=score,
        unmethylated_fraction=score,
        rss_quality=score,
        calibration_id="measured-track-calibration-v1",
    )


def test_complete_prior_is_stage_and_locus_gated():
    prior = AlphaGenome2VDJPrior(
        AlphaGenome2Status.COMPLETE, "pro-B", "IGH", "local-pro-B",
        conditioning_cell_count=32, input_sha256="a", output_sha256="b",
        evidence=(_evidence(),),
    )
    prior.validate()
    assert prior.segment_weights()["IGHV1-2*01"] > 1.0

    wrong = AlphaGenome2VDJPrior(
        AlphaGenome2Status.COMPLETE, "mature B", "IGH", "mature",
        conditioning_cell_count=32, input_sha256="a", output_sha256="b",
        evidence=(_evidence(),),
    )
    with pytest.raises(ValueError, match="not rearranged"):
        wrong.validate()


def test_noncommercial_outputs_cannot_silently_become_training_features():
    prior = AlphaGenome2VDJPrior(
        AlphaGenome2Status.COMPLETE, "pro-B", "IGH", "local-pro-B",
        conditioning_cell_count=32, input_sha256="a", output_sha256="b",
        evidence=(_evidence(),), training_use_allowed=False,
    )
    with pytest.raises(PermissionError):
        prior.as_training_features()


def test_unavailable_backend_never_fabricates_tracks():
    run = UnavailableAlphaGenome2Backend().build_prior(
        stage="pro-B", locus="IGH", intervals=[], raw_umi_counts=None,
        condition_name="local-pro-B",
    )
    run.validate()
    assert run.status is AlphaGenome2Status.UNAVAILABLE
    assert run.evidence == ()
    assert run.segment_weights() == {}


def test_regulatory_weights_change_only_segment_choice():
    recombinator = ChainRecombinator("IGH", rng=np.random.default_rng(7))
    target = recombinator.v_genes[0]
    weights = {gene: (1.0 if gene == target else 0.0) for gene in recombinator.v_genes}
    draws = [recombinator.rearrange(segment_weights=weights) for _ in range(20)]
    draws = [draw for draw in draws if draw is not None]
    assert draws and all(draw.v_gene == target for draw in draws)
    assert any(draw.junction.n_added >= 0 for draw in draws)


def test_empty_prior_exactly_reproduces_baseline_draw():
    a = ChainRecombinator("IGH", rng=np.random.default_rng(19))
    b = ChainRecombinator("IGH", rng=np.random.default_rng(19))
    assert a.rearrange().to_dict() == b.rearrange(segment_weights={}).to_dict()

