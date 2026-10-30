"""Class-switch recombination.

The same enzyme that drives somatic hypermutation, AID, also cuts the switch
regions upstream of the heavy-chain constant genes, swapping IgM for a
downstream isotype.  The variable region — and therefore the specificity —
is untouched; what changes is half-life and effector function.

Switching is modelled as a cytokine-conditioned categorical draw, since the
isotype distribution of a response is set by the helper cytokine milieu
rather than by anything in the BCR sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: Approximate serum half-life in days, used by the antibody output module.
ISOTYPE_HALF_LIFE_DAYS = {
    "IgM": 5.0, "IgD": 3.0, "IgG1": 21.0, "IgG2": 20.0, "IgG3": 7.0,
    "IgG4": 21.0, "IgA1": 6.0, "IgA2": 6.0, "IgE": 2.0,
}

#: Default switching propensities for a protein antigen in adjuvant.
DEFAULT_SWITCH_WEIGHTS = {
    "IgM": 0.18, "IgG1": 0.44, "IgG2": 0.12, "IgG3": 0.10,
    "IgG4": 0.04, "IgA1": 0.09, "IgE": 0.03,
}


@dataclass
class SwitchEvent:
    from_isotype: str
    to_isotype: str
    round_index: int


class ClassSwitchModule:
    """AID-dependent isotype switching inside the germinal centre."""

    def __init__(
        self,
        switch_weights: Optional[dict[str, float]] = None,
        switch_probability_per_round: float = 0.12,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        w = dict(switch_weights or DEFAULT_SWITCH_WEIGHTS)
        w.pop("IgM", None)
        total = sum(w.values())
        self.targets = list(w)
        self.probs = np.array([w[k] / total for k in self.targets])
        self.p_switch = switch_probability_per_round
        self.rng = rng if rng is not None else np.random.default_rng()

    def maybe_switch(self, isotype: str, round_index: int) -> Optional[SwitchEvent]:
        """Switching is irreversible and only moves away from IgM/IgD."""
        if isotype not in ("IgM", "IgD"):
            return None
        if self.rng.random() >= self.p_switch:
            return None
        new = self.targets[int(self.rng.choice(len(self.targets), p=self.probs))]
        return SwitchEvent(isotype, new, round_index)


__all__ = ["ClassSwitchModule", "SwitchEvent", "ISOTYPE_HALF_LIFE_DAYS",
           "DEFAULT_SWITCH_WEIGHTS"]
