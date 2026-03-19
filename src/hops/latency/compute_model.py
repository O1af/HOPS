"""Per-stage computation time modeling."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
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

        base_ms = (compute_ms + memory_ms) * self.latency_scale
        return max(0.0, base_ms + self.jitter.sample(rng))


class ComputeModel:
    """Maps each pipeline stage to a latency model."""

    def __init__(self, stage_models: dict[int, StageLatencySource],
                 backward_factor: float = 2.0,
                 backward_b_fraction: float = 0.5,
                 precision_speedup: float = 1.0):
        self._models = stage_models
        self._backward_factor = backward_factor
        self._backward_b_fraction = backward_b_fraction
        self._precision_speedup = precision_speedup

    def sample(self, stage_id: int, phase: Phase, rng: np.random.Generator) -> float:
        base = self._models[stage_id].sample(rng)
        if phase == Phase.BACKWARD:
            base *= self._backward_factor
        elif phase == Phase.BACKWARD_B:
            base *= self._backward_factor * self._backward_b_fraction
        elif phase == Phase.BACKWARD_W:
            base *= self._backward_factor * (1.0 - self._backward_b_fraction)
        if self._precision_speedup != 1.0:
            base /= self._precision_speedup
        return base

    @staticmethod
    def _stage_model_from_yaml(stage_cfg: dict, topology: Topology | None) -> StageLatencySource:
        """Legacy compatibility constructor for pre-canonical configs."""
        penalty_scale = 1.0
        penalty_offset_ms = 0.0
        memory_penalty_scale = 1.0
        memory_penalty_latency_us = 0.0
        if topology is not None:
            memory_placement = None
            if "memory_device_id" in stage_cfg:
                memory_placement = SimpleNamespace(
                    kind="device",
                    device=stage_cfg["memory_device_id"],
                )
            elif "memory_socket_id" in stage_cfg or "memory_node_id" in stage_cfg:
                memory_placement = SimpleNamespace(
                    kind="socket",
                    node=stage_cfg.get("memory_node_id"),
                    socket=stage_cfg.get("memory_socket_id"),
                )
            penalty = topology.stage_locality_penalty(
                device_id=stage_cfg["device"],
                memory_placement=memory_placement,
            )
            penalty_scale = penalty.compute_scale
            penalty_offset_ms = penalty.memory_latency_us / 1000.0
            memory_penalty_scale = penalty.memory_bandwidth_scale
            memory_penalty_latency_us = penalty.memory_latency_us

        if "compute_latency" in stage_cfg:
            return DistributionLatency(
                Distribution.from_yaml(stage_cfg["compute_latency"]),
                scale=penalty_scale,
                offset_ms=penalty_offset_ms if stage_cfg.get("memory_access_mb", 0.0) > 0 else 0.0,
            )

        if "compute_workload_tflop" not in stage_cfg:
            raise ValueError(
                f"Stage {stage_cfg['id']} must define either 'compute_latency' "
                "or 'compute_workload_tflop'"
            )

        if topology is None:
            raise ValueError(
                "Derived latency requires a topology with device capabilities"
            )

        device = topology.device(stage_cfg["device"])
        if device.flops is None or device.flops <= 0:
            raise ValueError(
                f"Stage {stage_cfg['id']} uses derived latency but device "
                f"{device.id!r} does not define a positive 'flops' value"
            )

        memory_access_mb = stage_cfg.get("memory_access_mb", 0.0)
        memory_bandwidth = device.memory_bandwidth_gbps
        if memory_access_mb > 0 and (memory_bandwidth is None or memory_bandwidth <= 0):
            raise ValueError(
                f"Stage {stage_cfg['id']} uses derived memory access but device "
                f"{device.id!r} does not define a positive "
                "'memory_bandwidth_gbps' value"
            )

        return DerivedLatency(
            workload_tflop=stage_cfg["compute_workload_tflop"],
            device_flops=device.flops,
            memory_access_mb=memory_access_mb,
            memory_bandwidth_gbps=memory_bandwidth or float("inf"),
            efficiency=stage_cfg.get("compute_efficiency", 1.0),
            memory_efficiency=stage_cfg.get("memory_efficiency", 1.0),
            compute_scale=penalty_scale,
            memory_bandwidth_scale=memory_penalty_scale,
            memory_latency_us=memory_penalty_latency_us,
            latency_scale=stage_cfg.get("latency_scale", 1.0),
            jitter=Distribution.from_yaml(
                stage_cfg.get("latency_jitter", {"type": "constant", "value": 0.0})
            ),
        )

    @classmethod
    def from_yaml(cls, config: dict, topology: Topology | None = None) -> "ComputeModel":
        """Legacy compatibility constructor for pre-canonical configs."""
        stage_models: dict[int, StageLatencySource] = {}
        for stage_cfg in config["stages"]:
            stage_models[stage_cfg["id"]] = cls._stage_model_from_yaml(stage_cfg, topology)
        return cls(
            stage_models,
            config.get("backward_factor", 2.0),
            config.get("backward_b_fraction", 0.5),
            config.get("precision_speedup", 1.0),
        )

    @staticmethod
    def _stage_model_from_config(stage: StageConfig, topology: Topology) -> StageLatencySource:
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

        return DerivedLatency(
            workload_tflop=stage.analytical.tflop,
            device_flops=device.flops,
            memory_access_mb=stage.analytical.memory_mb,
            memory_bandwidth_gbps=memory_bandwidth or float("inf"),
            efficiency=stage.analytical.efficiency_compute,
            memory_efficiency=stage.analytical.efficiency_memory,
            compute_scale=penalty.compute_scale,
            memory_bandwidth_scale=penalty.memory_bandwidth_scale,
            memory_latency_us=penalty.memory_latency_us,
            latency_scale=1.0,
            jitter=Distribution.from_yaml(stage.analytical.jitter),
        )

    @classmethod
    def from_pipeline_config(cls, pipeline: PipelineConfig,
                             topology: Topology) -> "ComputeModel":
        stage_models: dict[int, StageLatencySource] = {}
        for stage in pipeline.stages:
            stage_models[stage.id] = cls._stage_model_from_config(stage, topology)
        return cls(
            stage_models,
            backward_factor=pipeline.backward_factor,
            backward_b_fraction=pipeline.backward_split.activation_grad_fraction,
            precision_speedup=pipeline.precision.compute_speedup,
        )
