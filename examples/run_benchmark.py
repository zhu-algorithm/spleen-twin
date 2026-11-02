"""Small deterministic end-to-end benchmark used by the release checklist."""

from dataclasses import asdict
import json

from spleen_twin import Spleen
from spleen_twin.candidates import card_from_cell
from spleen_twin.modules.antigen import hen_egg_lysozyme
from spleen_twin.modules.developability import DevelopabilityPanel


def main():
    antigen = hen_egg_lysozyme()
    spleen = Spleen(seed=20261030, n_germinal_centers=2,
                    gc_kwargs={"carrying_capacity": 50, "max_light_zone_cells": 150})
    spleen.generate_naive_repertoire(150)
    result = spleen.immunise(antigen, n_rounds=3, max_founders=30)
    cells = sorted(result.plasma_cells + result.memory_cells, key=lambda c: c.log10_kd)[:5]
    panel = DevelopabilityPanel(apc_module=spleen.apc)
    cards = [card_from_cell(c, spleen.oracle, antigen.immunodominant().sequence, panel).to_dict()
             for c in cells]
    output = {"schema_version": "benchmark-1", "seed": spleen.seed_value,
              "repertoire": asdict(result.repertoire), "uncertainty": result.uncertainty,
              "organ_state": result.organ_state.to_dict(), "candidates": cards}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
