"""spleen-twin — a digital-twin spleen for antibody generation and maturation.

The package models the humoral response end to end: antigen presentation,
CD4+ T-cell help, V(D)J recombination, tolerance, B-cell activation,
germinal-centre affinity maturation, class switching, antibody output and
developability gating — and wraps them in a wet-dry closed loop.
"""

__version__ = "0.2.0"

from .organ import Spleen, CampaignResult, RepertoireStats, BCellStage, BoneMarrowTrajectory
from .modules import (affinity, ai_maturation, alphagenome_vdj, antigen,
                      developability, germinal_center, shm, vdj)
from .modules.receptor import PairedReceptor, EditingEvent
from .provenance import EvidenceClass, ParameterRecord, RunManifest

__all__ = ["Spleen", "CampaignResult", "RepertoireStats", "__version__",
           "BCellStage", "BoneMarrowTrajectory", "affinity", "ai_maturation",
           "alphagenome_vdj", "antigen", "developability", "germinal_center", "shm", "vdj",
           "PairedReceptor", "EditingEvent", "EvidenceClass", "ParameterRecord",
           "RunManifest"]
