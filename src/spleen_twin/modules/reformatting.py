"""Reformatting — from the in-vivo BCR to an expression-ready human IgG.

Platforms differ in how much of the constant region they humanise in vivo,
and that choice decides whether an in-vitro reformatting step is needed at
all.  Three designs are represented:

* **Reverse-chimeric (constant region stays murine).**  Human V, D and J
  segments sit in the mouse locus while the constant region remains
  murine, so the BCR still signals through the mouse Ig-alpha/Ig-beta
  sheath and class-switches normally.  The variable region is harvested
  and cloned onto a human constant region after the fact.  Used by
  VelocImmune, Kymouse, RenMab, OmniRat and AlivaMab (the platforms
  listed in :data:`REVERSE_CHIMERIC_PLATFORMS`).

* **Fully humanised constant region in vivo.**  XenoMouse carries
  megabase human Ig loci in which the mouse constant region is itself
  replaced by human, so the animal makes fully human antibodies directly
  and no constant-region swap is required.

* **Spatiotemporal switch.**  CAMouse uses a chimeric IgM BCR (human
  V(D)J on mouse C-mu) during development and switches through S-mu to a
  human C-gamma at the plasma-cell stage, secreting fully human IgG in
  vivo without post-hoc humanisation.

``ReformattingModule`` therefore treats the constant-region swap as a
*platform property*, not a universal step: see
:data:`PLATFORM_REQUIRES_CONSTANT_SWAP`.

Two engineering consequences are also represented:

* **Germline back-mutation.**  Framework replacement mutations accumulated
  during maturation raise immunogenicity risk and can destabilise the
  domain without contributing to binding.  Reverting them toward germline
  is routine; the module reports which positions are candidates and can
  apply the reversion.

* **ADAM-6.**  The mouse *Adam6* fertility genes lie inside the IgH V
  region.  Replacing that region can delete them, causing male sterility;
  platforms either re-insert an *Adam6* transgene or design the
  replacement to leave the locus intact.  This is a property of the
  platform, not of any one antibody, so it is recorded as a flag on the
  construct for provenance rather than simulated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

#: Human IgG1 CH1-hinge-CH2-CH3 is represented by name only; the twin
#: reasons about the variable domain, and the constant region is a
#: cassette choice rather than something the germinal centre evolves.
HUMAN_CONSTANT_REGIONS = ("IgG1", "IgG2", "IgG4", "IgA1", "IgM")

#: Platforms that keep a murine constant region in vivo and therefore need
#: the variable region cloned onto a human constant region after harvest.
REVERSE_CHIMERIC_PLATFORMS = (
    "VelocImmune", "Kymouse", "RenMab", "OmniRat", "AlivaMab",
)

#: Whether a named platform requires an in-vitro constant-region swap.
#: XenoMouse humanises the constant region in the locus itself; CAMouse
#: switches to a human constant region in vivo at the plasma-cell stage.
PLATFORM_REQUIRES_CONSTANT_SWAP = {
    "VelocImmune": True,
    "Kymouse": True,
    "RenMab": True,
    "OmniRat": True,
    "AlivaMab": True,
    "XenoMouse": False,
    "CAMouse": False,
    "generic-reverse-chimeric": True,
}

#: Platforms whose V-region replacement can collaterally delete the mouse
#: Adam6 fertility genes, which sit inside the IgH V region.
PLATFORM_ADAM6_RESCUE = {
    "VelocImmune": True, "Kymouse": False, "RenMab": True,
    "OmniRat": False, "AlivaMab": False, "XenoMouse": False,
    "CAMouse": False, "generic-reverse-chimeric": True,
}


@dataclass
class Construct:
    """A reformatted, expression-ready antibody."""

    name: str
    vh_aa: str
    vl_aa: Optional[str]
    constant_region: str
    germline_v: str
    n_v_mutations: int
    back_mutations_applied: list = field(default_factory=list)
    platform: str = "generic-reverse-chimeric"
    constant_swap_performed: bool = True
    adam6_rescue_required: bool = True
    notes: str = ""

    @property
    def humanness(self) -> float:
        """Fraction of the V domain that is unmutated human germline."""
        if not self.vh_aa:
            return 0.0
        return max(0.0, 1.0 - self.n_v_mutations / len(self.vh_aa))


class ReformattingModule:
    """Converts a matured B-cell receptor into a human IgG construct."""

    def __init__(
        self,
        constant_region: str = "IgG1",
        platform: str = "generic-reverse-chimeric",
        back_mutate_framework: bool = True,
    ) -> None:
        if constant_region not in HUMAN_CONSTANT_REGIONS:
            raise ValueError(f"unknown constant region {constant_region!r}")
        self.constant_region = constant_region
        self.platform = platform
        self.back_mutate_framework = back_mutate_framework

    def framework_back_mutation_candidates(self, cell) -> list:
        """Framework replacement mutations that are candidates for reversion."""
        return [
            m for m in cell.mutations
            if m.replacement and m.region.startswith("fwr")
        ]

    def reformat(self, cell, name: str, light_chain_aa: Optional[str] = None) -> Construct:
        """Clone the matured variable domain onto a human constant region."""
        germline_nt = cell.rearrangement.seq_nt
        seq_nt = cell.seq_nt
        applied = []

        if self.back_mutate_framework:
            chars = list(seq_nt)
            for m in self.framework_back_mutation_candidates(cell):
                if m.position < len(chars):
                    chars[m.position] = m.from_base
                    applied.append(m.label())
            seq_nt = "".join(chars)

        from .vdj import translate
        vh_aa = translate(seq_nt)

        remaining = sum(
            1 for m in cell.mutations
            if not (self.back_mutate_framework and m.replacement and m.region.startswith("fwr"))
        )
        swap = PLATFORM_REQUIRES_CONSTANT_SWAP.get(self.platform, True)
        notes = (
            "variable region matured in vivo on a murine constant region; "
            "constant region swapped to human after harvest"
            if swap else
            "platform produces a human constant region in vivo; no "
            "constant-region swap performed"
        )
        return Construct(
            name=name,
            vh_aa=vh_aa,
            vl_aa=light_chain_aa,
            constant_region=self.constant_region,
            germline_v=cell.rearrangement.v_gene,
            n_v_mutations=remaining,
            back_mutations_applied=applied,
            platform=self.platform,
            constant_swap_performed=swap,
            adam6_rescue_required=PLATFORM_ADAM6_RESCUE.get(self.platform, True),
            notes=notes,
        )


__all__ = ["ReformattingModule", "Construct", "HUMAN_CONSTANT_REGIONS",
           "REVERSE_CHIMERIC_PLATFORMS", "PLATFORM_REQUIRES_CONSTANT_SWAP",
           "PLATFORM_ADAM6_RESCUE"]
