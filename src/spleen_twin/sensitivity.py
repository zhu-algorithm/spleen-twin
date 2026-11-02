"""Small, replayable perturbation and sensitivity experiments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

from .organ.dynamics import aggregate_replicates


@dataclass(frozen=True)
class SweepPoint:
    parameters: dict
    seeds: tuple[int, ...]
    summary: dict


def parameter_sweep(runner: Callable[[dict, int], float], grid: dict[str, list],
                    seeds: tuple[int, ...] = (0, 1, 2)) -> list[SweepPoint]:
    """Evaluate every grid point with identical seeds and uncertainty summary."""
    names = tuple(grid)
    out = []
    for values in product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        observed = [float(runner(params, seed)) for seed in seeds]
        out.append(SweepPoint(params, seeds, aggregate_replicates(observed)))
    return out


def monotonic_effect(points: list[SweepPoint], parameter: str) -> dict:
    """Describe rather than overclaim the rank direction of a 1-D sweep."""
    from scipy.stats import spearmanr

    ordered = sorted(points, key=lambda p: p.parameters[parameter])
    x = [float(p.parameters[parameter]) for p in ordered]
    y = [float(p.summary["mean"]) for p in ordered]
    rho = float(spearmanr(x, y).statistic) if len(x) > 1 else float("nan")
    return {"parameter": parameter, "spearman": rho, "n_points": len(x),
            "direction": "increasing" if rho > 0 else "decreasing" if rho < 0 else "flat"}


__all__ = ["SweepPoint", "parameter_sweep", "monotonic_effect"]
