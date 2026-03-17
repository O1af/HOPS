"""Per-stage computation time modeling."""

import numpy as np

from hops.core.types import Phase
from hops.latency.distributions import Distribution


class ComputeModel:
    """Maps each pipeline stage to a latency distribution."""

    def __init__(self, stage_distributions: dict[int, Distribution],
                 backward_factor: float = 2.0,
                 backward_b_fraction: float = 0.5,
                 precision_speedup: float = 1.0):
        self._dists = stage_distributions
        self._backward_factor = backward_factor
        self._backward_b_fraction = backward_b_fraction
        self._precision_speedup = precision_speedup

    def sample(self, stage_id: int, phase: Phase, rng: np.random.Generator) -> float:
        base = self._dists[stage_id].sample(rng)
        if phase == Phase.BACKWARD:
            base *= self._backward_factor
        elif phase == Phase.BACKWARD_B:
            base *= self._backward_factor * self._backward_b_fraction
        elif phase == Phase.BACKWARD_W:
            base *= self._backward_factor * (1.0 - self._backward_b_fraction)
        if self._precision_speedup != 1.0:
            base /= self._precision_speedup
        return base

    @classmethod
    def from_yaml(cls, config: dict) -> "ComputeModel":
        dists = {}
        for stage_cfg in config["stages"]:
            dists[stage_cfg["id"]] = Distribution.from_yaml(stage_cfg["compute_latency"])
        return cls(
            dists,
            config.get("backward_factor", 2.0),
            config.get("backward_b_fraction", 0.5),
            config.get("precision_speedup", 1.0),
        )
