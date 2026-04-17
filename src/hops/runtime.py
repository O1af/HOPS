"""Runtime assembly for validated HOPS configs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hops.config import AppConfig, DeviceSpec, LinkOverride, OutputConfig
from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import make_scheduler, max_in_flight_count
from hops.failure.engine import FailureEngine
from hops.hardware.device import Device, numa_from_socket
from hops.hardware.network import Link
from hops.hardware.topology import LinkProfile, Locality, LocalityPenalty, Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Distribution
from hops.metrics.collector import MetricsCollector
from hops.metrics.reporter import Reporter
from hops.presets import PresetRegistry


@dataclass
class SimulationRuntime:
    engine: EventEngine
    pipeline: Pipeline
    collector: MetricsCollector
    reporter: Reporter
    output_config: OutputConfig
    num_batches: int
    num_microbatches: int


def _resolve_device(spec: DeviceSpec, overrides: dict[str, object], registry: PresetRegistry) -> Device:
    preset = registry.device(spec.preset)
    if preset.kind != spec.kind:
        raise ValueError(
            f"Device {spec.id!r} declares {spec.kind!r} preset usage but "
            f"preset {spec.preset!r} is of kind {preset.kind!r}"
        )
    override = overrides.get(spec.id)
    memory_mb = override.memory_mb if override and override.memory_mb is not None else preset.memory_mb
    flops_tflops = (
        override.flops_tflops
        if override and override.flops_tflops is not None
        else preset.flops_tflops
    )
    memory_bandwidth_gbps = (
        override.memory_bandwidth_gbps
        if override and override.memory_bandwidth_gbps is not None
        else preset.memory_bandwidth_gbps
    )
    numa_node = numa_from_socket(spec.socket)
    return Device(
        id=spec.id,
        kind=preset.kind,
        memory_mb=memory_mb,
        flops=flops_tflops,
        memory_bandwidth_gbps=memory_bandwidth_gbps,
        node_id=spec.node,
        socket_id=spec.socket,
        numa_node=numa_node,
    )


def _resolve_link_profiles(config: AppConfig, registry: PresetRegistry
                           ) -> tuple[dict[Locality, LinkProfile], dict[Locality, LocalityPenalty]]:
    interconnect = config.hardware.interconnect
    presets = {
        Locality.SAME_SOCKET: registry.interconnect(interconnect.same_socket or interconnect.same_node),
        Locality.SAME_NODE: registry.interconnect(interconnect.same_node),
        Locality.CROSS_NODE: registry.interconnect(interconnect.cross_node),
    }
    profiles = {
        locality: LinkProfile(
            bandwidth_gbps=preset.bandwidth_gbps,
            base_latency_us=preset.latency_us,
            jitter=Distribution.from_yaml(preset.jitter),
        )
        for locality, preset in presets.items()
    }
    penalties = {locality: preset.penalty for locality, preset in presets.items()}
    return profiles, penalties


def _resolve_link_overrides(link_overrides: list[LinkOverride]) -> list[Link]:
    return [
        Link(
            src=override.src,
            dst=override.dst,
            bandwidth_gbps=override.bandwidth_gbps,
            base_latency_us=override.latency_us,
            jitter=Distribution.from_yaml(override.jitter),
        )
        for override in link_overrides
    ]


def _resolve_activation_mb(config: AppConfig) -> float:
    """Return the fp32-equivalent activation size in MB.

    If the user supplied ``activation_mb`` explicitly it is returned as-is.
    Otherwise it is derived from the ``model`` block:
    ``hidden_dim * seq_len * 4 bytes / (1024 * 1024)``.
    Precision scaling is applied separately downstream. This derivation is a
    convenience heuristic; experiment-calibrated configs should continue to
    provide ``activation_mb`` explicitly.
    """
    if config.pipeline.activation_mb is not None:
        return config.pipeline.activation_mb
    model = config.pipeline.model
    assert model is not None  # enforced by config parser
    return model.hidden_dim * model.seq_len * 4 / (1024 * 1024)


def validate_memory(config: AppConfig, topology: Topology, activation_mb: float) -> None:
    eff_activation = activation_mb * config.pipeline.precision.data_scale
    weight_overhead = config.pipeline.precision.weight_memory_overhead
    usage_by_device: dict[str, dict[str, float]] = {}

    for stage in config.pipeline.stages:
        device = topology.device(stage.device)
        usage = usage_by_device.setdefault(
            device.id,
            {"weights_mb": 0.0, "activations_mb": 0.0},
        )
        usage["weights_mb"] += stage.weights_mb * weight_overhead
        usage["activations_mb"] += (
            eff_activation
            * max_in_flight_count(
                config.pipeline.schedule,
                stage.id,
                len(config.pipeline.stages),
                config.simulation.microbatches,
            )
        )

    for device_id, usage in usage_by_device.items():
        device = topology.device(device_id)
        peak = usage["weights_mb"] + usage["activations_mb"]
        if peak > device.memory_mb:
            raise ValueError(
                f"Device {device.id}: peak memory {peak:.1f} MB exceeds device capacity "
                f"{device.memory_mb:.1f} MB (weights={usage['weights_mb']:.1f} MB, "
                f"activations={usage['activations_mb']:.1f} MB)"
            )


def build_runtime(config: AppConfig, registry: PresetRegistry | None = None) -> SimulationRuntime:
    registry = registry or PresetRegistry()
    rng = np.random.default_rng(config.simulation.seed)
    device_overrides = {override.id: override for override in config.overrides.devices}
    devices = [_resolve_device(spec, device_overrides, registry) for spec in config.hardware.devices]
    profiles, penalties = _resolve_link_profiles(config, registry)
    topology = Topology(
        devices=devices,
        links=_resolve_link_overrides(config.overrides.links),
        link_profiles=profiles,
        locality_penalties=penalties,
    )
    activation_mb = _resolve_activation_mb(config)
    validate_memory(config, topology, activation_mb)

    compute_model = ComputeModel.from_pipeline_config(config.pipeline, topology=topology)
    scheduler = make_scheduler({"policy": config.pipeline.schedule})
    collector = MetricsCollector()
    engine = EventEngine()
    stages = [Stage(id=stage.id, device_id=stage.device) for stage in config.pipeline.stages]

    optimizer_latency = None
    if config.optimizer.enabled and config.optimizer.update_distribution is not None:
        optimizer_latency = Distribution.from_yaml(config.optimizer.update_distribution)

    pipeline = Pipeline(
        stages=stages,
        engine=engine,
        topology=topology,
        compute_model=compute_model,
        scheduler=scheduler,
        collector=collector,
        activation_size_mb=activation_mb,
        rng=rng,
        optimizer_latency=optimizer_latency,
        gradient_size_mb=config.optimizer.gradient_mb,
        stage_memory_mb={stage.id: stage.weights_mb for stage in config.pipeline.stages},
        gradient_accumulation_steps=config.optimizer.accumulation_steps,
        precision=config.pipeline.precision,
        allreduce_algo=config.optimizer.allreduce_algorithm,
    )

    if config.failure.enabled:
        pipeline.set_failure_engine(FailureEngine(
            engine=engine,
            topology=topology,
            collector=collector,
            config=config.failure,
            rng=rng,
        ))

    reporter = Reporter(collector)
    return SimulationRuntime(
        engine=engine,
        pipeline=pipeline,
        collector=collector,
        reporter=reporter,
        output_config=config.output,
        num_batches=config.simulation.batches,
        num_microbatches=config.simulation.microbatches,
    )
