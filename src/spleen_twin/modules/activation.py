"""B-cell activation at the T-B border.

A naive B cell that captures antigen with its BCR, processes it and presents
it to a primed Tfh cell receives help and commits to one of two fates: an
extrafollicular plasmablast focus giving fast, low-affinity antibody, or a
germinal centre giving slow, high-affinity antibody.  The split is
affinity-dependent, which is why early serum antibody is poor and late serum
antibody is good.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class ActivationFate(str, Enum):
    IGNORED = "ignored"
    EXTRAFOLLICULAR = "extrafollicular"
    GERMINAL_CENTER = "germinal_center"


@dataclass
class ActivationResult:
    fate: ActivationFate
    bcr_signal: float
    help_received: float


class ActivationModule:
    """Decides which naive B cells enter a response and where they go."""

    def __init__(
        self,
        activation_threshold_log10_kd: float = -5.5,
        gc_entry_bias: float = 0.65,
        signal_midpoint_log10_kd: float = -6.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.activation_threshold = activation_threshold_log10_kd
        self.gc_entry_bias = gc_entry_bias
        self.signal_midpoint = signal_midpoint_log10_kd
        self.rng = rng if rng is not None else np.random.default_rng()

    def bcr_signal(self, log10_kd: float) -> float:
        """Occupancy-like signal strength, saturating in affinity."""
        return float(1.0 / (1.0 + 10.0 ** (log10_kd - self.signal_midpoint)))

    def activate(self, log10_kd: float, tfh: Optional[object] = None,
                 help_available: float = 1.0) -> ActivationResult:
        if log10_kd > self.activation_threshold:
            return ActivationResult(ActivationFate.IGNORED, 0.0, 0.0)
        sig = self.bcr_signal(log10_kd)
        help_recv = sig * help_available
        p_gc = self.gc_entry_bias * sig
        fate = (ActivationFate.GERMINAL_CENTER
                if self.rng.random() < p_gc else ActivationFate.EXTRAFOLLICULAR)
        return ActivationResult(fate, sig, help_recv)


__all__ = ["ActivationModule", "ActivationResult", "ActivationFate"]
