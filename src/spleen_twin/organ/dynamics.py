"""Explicit compartment, event-clock and uncertainty contracts for the twin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class Compartment(str, Enum):
    RED_PULP = "red_pulp"
    MARGINAL_ZONE = "marginal_zone"
    PALS = "pals"
    FOLLICLE = "follicle"
    GC_DARK = "gc_dark_zone"
    GC_LIGHT = "gc_light_zone"
    PLASMA = "plasma"
    MEMORY = "memory"


@dataclass(frozen=True)
class Transition:
    source: Compartment
    target: Compartment
    event: str
    timescale_hours: float
    evidence: str


TRANSITIONS = (
    Transition(Compartment.RED_PULP, Compartment.MARGINAL_ZONE, "antigen_capture", 1.0, "literature"),
    Transition(Compartment.MARGINAL_ZONE, Compartment.PALS, "antigen_presentation", 12.0, "literature"),
    Transition(Compartment.PALS, Compartment.FOLLICLE, "t_b_conjugate", 24.0, "literature"),
    Transition(Compartment.FOLLICLE, Compartment.GC_DARK, "gc_seeding", 72.0, "literature"),
    Transition(Compartment.GC_DARK, Compartment.GC_LIGHT, "division_and_shm", 8.0, "model"),
    Transition(Compartment.GC_LIGHT, Compartment.GC_DARK, "selection_and_recycling", 8.0, "model"),
    Transition(Compartment.GC_LIGHT, Compartment.PLASMA, "plasma_exit", 8.0, "model"),
    Transition(Compartment.GC_LIGHT, Compartment.MEMORY, "memory_exit", 8.0, "model"),
)


@dataclass
class OrganState:
    time_hours: float = 0.0
    antigen_nm: float = 0.0
    presented_antigen_nm: float = 0.0
    tfh_help: float = 0.0
    naive_b: int = 0
    gc_b: int = 0
    plasma_b: int = 0
    memory_b: int = 0

    def validate(self) -> None:
        if self.time_hours < 0 or self.antigen_nm < 0 or self.presented_antigen_nm < 0:
            raise ValueError("time and antigen quantities must be non-negative")
        if min(self.naive_b, self.gc_b, self.plasma_b, self.memory_b) < 0:
            raise ValueError("cell counts must be non-negative")
        if not 0.0 <= self.tfh_help <= 1.0:
            raise ValueError("tfh_help must be in [0, 1]")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def aggregate_replicates(values: Iterable[float]) -> dict:
    """Report distributional uncertainty without implying experimental CI."""
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "sd": None, "q05": None, "q50": None, "q95": None}
    return {
        "n": int(x.size), "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "q05": float(np.quantile(x, 0.05)), "q50": float(np.quantile(x, 0.5)),
        "q95": float(np.quantile(x, 0.95)),
    }


__all__ = ["Compartment", "Transition", "TRANSITIONS", "OrganState", "aggregate_replicates"]
