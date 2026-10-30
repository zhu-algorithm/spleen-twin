"""Atlas grounding — organ parameters measured, not assumed.

The composition of the virtual spleen is taken from real single-cell data
rather than chosen by hand.  The reference shipped with the package is the
whole human spleen slice of CELLxGENE Census (release 2025-11-08,
``tissue_general == 'spleen' and is_primary_data == True``): 577,677 cells
from 58 donors across 7 datasets, 110 annotated cell types, all of them
from tissue annotated ``normal``.

The two numbers that actually move the simulation are:

* **Tfh per naive B cell (~0.110).**  T-cell help is the limiting resource
  of the germinal centre, so this ratio sets how hard the competition in
  the light zone is.  Choosing it by hand is choosing the answer.
* **Memory to plasma ratio (~1.64).**  Used as a sanity check on the
  germinal-centre exit split.  It is a steady-state abundance, not a
  production rate — the two compartments have different lifespans — so it
  bounds the exit split rather than fixing it.

Germinal-centre B cells are only ~0.06% of a resting spleen, which is the
expected result: germinal centres are induced by immunisation, so a
resting-tissue atlas should contain almost none.  That is a useful
negative control on the atlas, not a defect in it.

To regenerate the reference, see ``scripts/fetch_spleen_census.py``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class AtlasReference:
    """Measured spleen composition used to parameterise the organ."""

    raw: dict

    @property
    def n_cells(self) -> int:
        return self.raw["n_cells"]

    @property
    def n_donors(self) -> int:
        return self.raw["n_donors"]

    @property
    def tfh_per_naive_b(self) -> float:
        return self.raw["derived_ratios"]["tfh_per_naive_b"]

    @property
    def memory_per_plasma(self) -> float:
        return self.raw["derived_ratios"]["memory_per_plasma"]

    @property
    def gc_fraction_of_b(self) -> float:
        return self.raw["derived_ratios"]["gc_fraction_of_b"]

    def fraction(self, compartment: str) -> float:
        return self.raw["fractions_of_spleen"][compartment]

    def gc_parameters(self, n_founders: int) -> dict:
        """Germinal-centre settings implied by the measured composition.

        ``tfh_help_slots`` is the number of centrocytes that can be rescued
        per cycle.  It is derived from the measured Tfh-to-naive-B ratio
        rather than tuned: with ~0.11 Tfh per naive B cell, a light zone
        holding ``n`` centrocytes has help for roughly ``0.11 n`` of them.
        """
        return {
            "tfh_help_slots": max(1, int(round(self.tfh_per_naive_b * n_founders))),
            "memory_fraction_of_exits": self.memory_per_plasma / (1.0 + self.memory_per_plasma),
        }

    def summary(self) -> dict:
        return {
            "source": self.raw["source"],
            "n_cells": self.n_cells,
            "n_donors": self.n_donors,
            "n_cell_types": self.raw["n_cell_types"],
            "tfh_per_naive_b": round(self.tfh_per_naive_b, 4),
            "memory_per_plasma": round(self.memory_per_plasma, 3),
            "b_fraction_of_spleen": round(self.fraction("b_total"), 4),
        }


def load_reference(path: Optional[str] = None) -> AtlasReference:
    """Load the packaged spleen composition reference."""
    with open(path or os.path.join(_DATA, "spleen_reference.json")) as fh:
        return AtlasReference(json.load(fh))


__all__ = ["AtlasReference", "load_reference"]
