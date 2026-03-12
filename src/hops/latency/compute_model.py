"""Per-stage computation time modeling."""

from hops.core.types import Phase
from hops.latency.distributions import Distribution


class ComputeModel:
    """Maps each pipeline stage to a latency distribution."""

    def __init__(self, stage_distributions: dict[int, Distribution],
                 backward_factor: float = 2.0):
        self._dists = stage_distributions
        self._backward_factor = backward_factor

    def sample(self, stage_id: int, phase: Phase) -> float:
        base = self._dists[stage_id].sample()
        if phase == Phase.BACKWARD:
            base *= self._backward_factor
        return base

    @classmethod
    def from_yaml(cls, config: dict) -> "ComputeModel":
        dists = {}
        for stage_cfg in config["stages"]:
            dists[stage_cfg["id"]] = Distribution.from_yaml(stage_cfg["compute_latency"])
        return cls(dists, config.get("backward_factor", 2.0))
