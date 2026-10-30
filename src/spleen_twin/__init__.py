"""spleen-twin — a digital-twin spleen for antibody generation and maturation.

The package models the humoral response end to end: antigen presentation,
CD4+ T-cell help, V(D)J recombination, tolerance, B-cell activation,
germinal-centre affinity maturation, class switching, antibody output and
developability gating — and wraps them in a wet-dry closed loop.
"""

__version__ = "0.1.0"

from .organ import Spleen, CampaignResult, RepertoireStats
from .modules import affinity, antigen, developability, germinal_center, shm, vdj

__all__ = ["Spleen", "CampaignResult", "RepertoireStats", "__version__",
           "affinity", "antigen", "developability", "germinal_center", "shm", "vdj"]
