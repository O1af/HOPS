"""Per-stage computation time modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hops.config import PipelineConfig, StageConfig
from hops.core.types import Phase
from hops.hardware.topology import Topology
from hops.latency.distributions import Constant, Distribution


class StageLatencySource(Protocol):
    def sample(self, rng: np.random.Generator) -> float:
        """Return a forward-pass latency sample in milliseconds."""


@dataclass
class DistributionLatency:
    distribution: Distribution
    scale: float = 1.0
    offset_ms: float = 0.0

    def sample(self, rng: np.random.Generator) -> float:
        return max(0.0, self.distribution.sample(rng) * self.scale + self.offset_ms)


# Fixed per-invocation overhead (kernel launch + Python/CUDA glue). For tiny
# transformer stages on modern GPUs this dwarfs the pure FLOP-budget estimate.
DEFAULT_LAUNCH_OVERHEAD_MS = 1.5


@dataclass
class DerivedLatency:
    workload_tflop: float
    device_flops: float
    memory_access_mb: float
    memory_bandwidth_gbps: float
    efficiency: float = 1.0
    memory_efficiency: float = 1.0
    compute_scale: float = 1.0
    memory_bandwidth_scale: float = 1.0
    memory_latency_us: float = 0.0
    latency_scale: float = 1.0
    launch_overhead_ms: float = DEFAULT_LAUNCH_OVERHEAD_MS
    jitter: Distribution = Constant(0.0)

    def sample(self, rng: np.random.Generator) -> float:
        effective_flops = self.device_flops * self.efficiency
        compute_ms = 1000.0 * self.workload_tflop / effective_flops
        compute_ms *= self.compute_scale

        memory_ms = 0.0
        if self.memory_access_mb > 0:
            effective_bw = (
                self.memory_bandwidth_gbps
                * self.memory_efficiency
                * self.memory_bandwidth_scale
            )
            memory_ms = (self.memory_access_mb * 8.0) / effective_bw
            memory_ms += self.memory_latency_us / 1000.0

        base_ms = (compute_ms + memory_ms) * self.latency_scale + self.launch_overhead_ms
        return max(0.0, base_ms + self.jitter.sample(rng))


class ComputeModel:
    """Maps each pipeline stage to a latency model."""

    def __init__(self, stage_models: dict[int, StageLatencySource],
                 backward_factor: float = 2.0,
                 backward_b_fraction: float = 0.5,
                 backward_models: dict[int, StageLatencySource] | None = None):
        self._models = stage_models
        self._backward_models = backward_models or {}
        self._backward_factor = backward_factor
        self._backward_b_fraction = backward_b_fraction

    def sample(self, stage_id: int, phase: Phase, rng: np.random.Generator) -> float:
        if phase in (Phase.BACKWARD, Phase.BACKWARD_B, Phase.BACKWARD_W) \
                and stage_id in self._backward_models:
            base = self._backward_models[stage_id].sample(rng)
            if phase == Phase.BACKWARD_B:
                base *= self._backward_b_fraction
            elif phase == Phase.BACKWARD_W:
                base *= 1.0 - self._backward_b_fraction
        else:
            base = self._models[stage_id].sample(rng)
            if phase == Phase.BACKWARD:
                base *= self._backward_factor
            elif phase == Phase.BACKWARD_B:
                base *= self._backward_factor * self._backward_b_fraction
            elif phase == Phase.BACKWARD_W:
                base *= self._backward_factor * (1.0 - self._backward_b_fraction)
        return base

    @staticmethod
    def _stage_model_from_config(stage: StageConfig, topology: Topology,
                                 precision_speedup: float = 1.0) -> StageLatencySource:
        penalty = topology.stage_locality_penalty(
            device_id=stage.device,
            memory_placement=stage.memory_placement,
        )
        if stage.compute_mode == "explicit":
            assert stage.explicit is not None
            return DistributionLatency(Distribution.from_yaml(stage.explicit.distribution))

        assert stage.analytical is not None
        device = topology.device(stage.device)
        if device.flops is None or device.flops <= 0:
            raise ValueError(
                f"Stage {stage.id} uses analytical compute but device {device.id!r} "
                "does not define a positive flops value"
            )
        memory_bandwidth = device.memory_bandwidth_gbps
        if stage.analytical.memory_mb > 0 and (memory_bandwidth is None or memory_bandwidth <= 0):
            raise ValueError(
                f"Stage {stage.id} uses analytical memory access but device {device.id!r} "
                "does not define a positive memory bandwidth value"
            )

        launch_overhead_ms = (
            device.launch_overhead_ms
            if device.launch_overhead_ms is not None
            else DEFAULT_LAUNCH_OVERHEAD_MS
        )
        return DerivedLatency(
            workload_tflop=stage.analytical.tflop,
            device_flops=device.flops,
            memory_access_mb=stage.analytical.memory_mb,
            memory_bandwidth_gbps=memory_bandwidth or float("inf"),
            efficiency=stage.analytical.efficiency_compute,
            memory_efficiency=stage.analytical.efficiency_memory,
            compute_scale=penalty.compute_scale / precision_speedup,
            memory_bandwidth_scale=penalty.memory_bandwidth_scale,
            memory_latency_us=penalty.memory_latency_us,
            latency_scale=1.0,
            launch_overhead_ms=launch_overhead_ms,
            jitter=Distribution.from_yaml(stage.analytical.jitter),
        )

    @classmethod
    def from_pipeline_config(cls, pipeline: PipelineConfig,
                             topology: Topology) -> "ComputeModel":
        precision_speedup = pipeline.precision.compute_speedup
        stage_models: dict[int, StageLatencySource] = {}
        backward_models: dict[int, StageLatencySource] = {}
        for stage in pipeline.stages:
            stage_models[stage.id] = cls._stage_model_from_config(
                stage, topology, precision_speedup=precision_speedup,
            )
            if stage.backward is not None:
                backward_models[stage.id] = DistributionLatency(
                    Distribution.from_yaml(stage.backward.distribution)
                )
        return cls(
            stage_models,
            backward_factor=pipeline.backward_factor,
            backward_b_fraction=pipeline.backward_split.activation_grad_fraction,
            backward_models=backward_models,
        )
