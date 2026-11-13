from spleen_twin.modules.ai_maturation import (
    PairedSequence, RegulatoryAwareMaturationModel, RegulatoryState,
    SequenceProposal, StructureAssessment,
)


class GeneModel:
    def encode(self, expression, regulatory_edges):
        return RegulatoryState((expression["AICDA"],), {"SHM": expression["AICDA"]})


class VDJModel:
    def propose(self, parent, state, antigen_sequence, n):
        return [SequenceProposal(parent, -1.0), SequenceProposal(
            PairedSequence(parent.heavy_aa + "Y", parent.light_aa), -2.0,
            changed_heavy_positions=(len(parent.heavy_aa),))]


class Diffusion:
    def refine(self, proposals, antigen_sequence, state):
        return proposals


class Structure:
    def assess(self, antibody, antigen_sequence):
        good = antibody.heavy_aa.endswith("Y")
        return StructureAssessment(0.9, 0.9 if good else 0.5, 0.0, provenance="test")


class Affinity:
    def predict(self, antibody, antigen_sequence, structure):
        return (-9.0 if antibody.heavy_aa.endswith("Y") else -7.0, 0.2)


def test_multimodal_stack_ranks_joint_sequence_structure_affinity():
    model = RegulatoryAwareMaturationModel(
        GeneModel(), VDJModel(), Diffusion(), Structure(), Affinity()
    )
    parent = PairedSequence("EVQL", "DIQM")
    ranked = model.propose_and_rank(parent, "ANTIGEN", {"AICDA": 1.0}, [("BCL6", "AICDA")])
    assert ranked[0].proposal.sequence.heavy_aa.endswith("Y")
    assert ranked[0].predicted_log10_kd == -9.0
    assert ranked[0].structure.provenance == "test"

