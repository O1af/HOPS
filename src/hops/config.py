"""Canonical HOPS configuration schema and parser."""

from __future__ import annotations

from dataclasses import dataclass, field

from hops.core.types import AllreduceAlgo, Precision
from hops.latency.distributions import Distribution


DEFAULT_COMPUTE_EFFICIENCY = 0.45
"""Fallback compute efficiency (matmul-heavy stages achieve ~50-60% of peak)."""

DEFAULT_MEMORY_EFFICIENCY = 0.6
"""Fallback memory bandwidth efficiency (memory-bound ops see ~60-80% utilization)."""


def _require_positive(value: float | int, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be > 0, got {value}")


def _require_non_negative(value: float | int, label: str) -> None:
    if value < 0:
        raise ValueError(f"{label} must be >= 0, got {value}")


def _require_distribution(config: dict, label: str) -> None:
    try:
        Distribution.from_yaml(config)
    except (ValueError, KeyError, TypeError) as exc:  # pragma: no cover - error message wrapper
        raise ValueError(f"Invalid distribution for {label}: {exc}") from exc


@dataclass(frozen=True)
class AnalyticalComputeConfig:
    tflop: float
    memory_mb: float = 0.0
    efficiency_compute: float = 1.0
    efficiency_memory: float = 1.0
    jitter: dict = field(default_factory=lambda: {"type": "constant", "value": 0.0})


@dataclass(frozen=True)
class ExplicitComputeConfig:
    distribution: dict


@dataclass(frozen=True)
class MemoryPlacement:
    kind: str = "local"
    node: str | None = None
    socket: str | int | None = None
    device: str | None = None


@dataclass(frozen=True)
class BackwardComputeConfig:
    """Explicit per-stage backward latency distribution.

    When present, ComputeModel uses this distribution directly for the BACKWARD
    phase instead of scaling forward by pipeline.backward_factor. ZeroBubble
    BACKWARD_B and BACKWARD_W are still derived from this via
    backward_split.activation_grad_fraction.
    """
    distribution: dict


@dataclass(frozen=True)
class StageConfig:
    id: int
    device: str
    weights_mb: float
    compute_mode: str
    analytical: AnalyticalComputeConfig | None = None
    explicit: ExplicitComputeConfig | None = None
    backward: BackwardComputeConfig | None = None
    memory_placement: MemoryPlacement = field(default_factory=MemoryPlacement)


@dataclass(frozen=True)
class BackwardSplitConfig:
    enabled: bool = False
    activation_grad_fraction: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    seq_len: int


@dataclass(frozen=True)
class PipelineConfig:
    schedule: str
    precision: Precision
    activation_mb: float | None
    backward_factor: float
    backward_split: BackwardSplitConfig
    stages: list[StageConfig]
    model: ModelConfig | None = None


@dataclass(frozen=True)
class SimulationConfig:
    batches: int
    microbatches: int
    seed: int


@dataclass(frozen=True)
class DeviceSpec:
    id: str
    preset: str
    kind: str
    node: str
    socket: str


@dataclass(frozen=True)
class InterconnectConfig:
    same_socket: str | None
    same_node: str
    cross_node: str


@dataclass(frozen=True)
class HardwareConfig:
    devices: list[DeviceSpec]
    interconnect: InterconnectConfig


@dataclass(frozen=True)
class OptimizerConfig:
    enabled: bool = False
    gradient_mb: float = 0.0
    accumulation_steps: int = 1
    allreduce_algorithm: AllreduceAlgo = AllreduceAlgo.NAIVE
    update_distribution: dict | None = None
    iteration_barrier: dict | None = None


@dataclass(frozen=True)
class FailureConfig:
    enabled: bool = False
    check_interval_ms: float = 10.0
    device_failure_probability: float = 0.001
    link_failure_probability: float = 0.0005
    recovery_time_ms: float = 5.0


@dataclass(frozen=True)
class OutputConfig:
    timeline: str | None = None
    dashboard: str | None = None
    summary_json: str | None = None
    trace_csv: str | None = None


@dataclass(frozen=True)
class DeviceOverride:
    id: str
    memory_mb: float | None = None
    flops_tflops: float | None = None
    memory_bandwidth_gbps: float | None = None
    launch_overhead_ms: float | None = None


@dataclass(frozen=True)
class LinkOverride:
    src: str
    dst: str
    bandwidth_gbps: float
    latency_us: float
    jitter: dict = field(default_factory=lambda: {"type": "constant", "value": 0.0})


@dataclass(frozen=True)
class OverridesConfig:
    devices: list[DeviceOverride] = field(default_factory=list)
    links: list[LinkOverride] = field(default_factory=list)


@dataclass(frozen=True)
class AppConfig:
    simulation: SimulationConfig
    pipeline: PipelineConfig
    hardware: HardwareConfig
    optimizer: OptimizerConfig
    failure: FailureConfig
    output: OutputConfig
    overrides: OverridesConfig


class ConfigParser:
    """Parse and validate the canonical preset-first HOPS schema."""

    REQUIRED_TOP_LEVEL = ("simulation", "pipeline", "hardware", "optimizer", "failure", "output")

    def parse(self, raw: dict) -> AppConfig:
        for section in self.REQUIRED_TOP_LEVEL:
            if section not in raw:
                raise ValueError(f"Missing required config section {section!r}")

        simulation = self._parse_simulation(raw["simulation"])
        pipeline = self._parse_pipeline(raw["pipeline"])
        hardware = self._parse_hardware(raw["hardware"])
        optimizer = self._parse_optimizer(raw["optimizer"])
        failure = self._parse_failure(raw["failure"])
        output = self._parse_output(raw["output"])
        overrides = self._parse_overrides(raw.get("overrides", {}))

        self._validate_stage_devices(pipeline, hardware)
        return AppConfig(
            simulation=simulation,
            pipeline=pipeline,
            hardware=hardware,
            optimizer=optimizer,
            failure=failure,
            output=output,
            overrides=overrides,
        )

    def _parse_simulation(self, raw: dict) -> SimulationConfig:
        _require_positive(raw["batches"], "simulation.batches")
        _require_positive(raw["microbatches"], "simulation.microbatches")
        return SimulationConfig(
            batches=raw["batches"],
            microbatches=raw["microbatches"],
            seed=raw["seed"],
        )

    def _parse_pipeline(self, raw: dict) -> PipelineConfig:
        activation_mb = raw.get("activation_mb")
        if activation_mb is not None:
            _require_non_negative(activation_mb, "pipeline.activation_mb")
        _require_positive(raw.get("backward_factor", 2.0), "pipeline.backward_factor")

        model = self._parse_model(raw.get("model"))

        if activation_mb is None and model is None:
            raise ValueError(
                "pipeline must specify either 'activation_mb' or a 'model' block "
                "(with hidden_dim and seq_len) so activation size can be derived"
            )

        split_raw = raw.get("backward_split", {})
        backward_split = BackwardSplitConfig(
            enabled=split_raw.get("enabled", False),
            activation_grad_fraction=split_raw.get("activation_grad_fraction", 0.5),
        )
        if not 0.0 <= backward_split.activation_grad_fraction <= 1.0:
            raise ValueError(
                "pipeline.backward_split.activation_grad_fraction must be in [0, 1]"
            )

        stages = [self._parse_stage(idx, stage_raw) for idx, stage_raw in enumerate(raw["stages"])]
        if not stages:
            raise ValueError("pipeline.stages must contain at least one stage")

        return PipelineConfig(
            schedule=raw["schedule"],
            precision=Precision(raw.get("precision", "fp32")),
            activation_mb=activation_mb,
            backward_factor=raw.get("backward_factor", 2.0),
            backward_split=backward_split,
            stages=stages,
            model=model,
        )

    def _parse_model(self, raw: dict | None) -> ModelConfig | None:
        if raw is None:
            return None
        if "hidden_dim" not in raw or "seq_len" not in raw:
            raise ValueError("pipeline.model requires 'hidden_dim' and 'seq_len'")
        _require_positive(raw["hidden_dim"], "pipeline.model.hidden_dim")
        _require_positive(raw["seq_len"], "pipeline.model.seq_len")
        return ModelConfig(hidden_dim=raw["hidden_dim"], seq_len=raw["seq_len"])

    def _parse_stage(self, stage_id: int, raw: dict) -> StageConfig:
        _require_non_negative(raw["weights_mb"], f"pipeline.stages[{stage_id}].weights_mb")
        if "compute" not in raw:
            raise ValueError(f"pipeline.stages[{stage_id}] must define a compute block")
        compute_raw = raw["compute"]
        mode = compute_raw["mode"]
        memory_placement = self._parse_memory_placement(stage_id, raw.get("memory_placement", {}))
        backward = self._parse_backward(stage_id, raw.get("backward"))

        if mode == "analytical":
            if "tflop" not in compute_raw:
                raise ValueError(
                    f"pipeline.stages[{stage_id}].compute with mode=analytical "
                    "requires tflop"
                )
            _require_positive(compute_raw["tflop"], f"pipeline.stages[{stage_id}].compute.tflop")
            _require_non_negative(
                compute_raw.get("memory_mb", 0.0),
                f"pipeline.stages[{stage_id}].compute.memory_mb",
            )
            efficiency = compute_raw.get("efficiency", {})
            analytical = AnalyticalComputeConfig(
                tflop=compute_raw["tflop"],
                memory_mb=compute_raw.get("memory_mb", 0.0),
                efficiency_compute=efficiency.get("compute", DEFAULT_COMPUTE_EFFICIENCY),
                efficiency_memory=efficiency.get("memory", DEFAULT_MEMORY_EFFICIENCY),
                jitter=compute_raw.get("jitter", {"type": "constant", "value": 0.0}),
            )
            _require_positive(
                analytical.efficiency_compute,
                f"pipeline.stages[{stage_id}].compute.efficiency.compute",
            )
            _require_positive(
                analytical.efficiency_memory,
                f"pipeline.stages[{stage_id}].compute.efficiency.memory",
            )
            _require_distribution(analytical.jitter, f"pipeline.stages[{stage_id}].compute.jitter")
            return StageConfig(
                id=stage_id,
                device=raw["device"],
                weights_mb=raw["weights_mb"],
                compute_mode=mode,
                analytical=analytical,
                backward=backward,
                memory_placement=memory_placement,
            )

        if mode == "explicit":
            if memory_placement.kind != "local":
                raise ValueError(
                    f"pipeline.stages[{stage_id}].memory_placement is only supported "
                    "for analytical compute mode"
                )
            if "distribution" not in compute_raw:
                raise ValueError(
                    f"pipeline.stages[{stage_id}].compute with mode=explicit "
                    "requires distribution"
                )
            _require_distribution(
                compute_raw["distribution"],
                f"pipeline.stages[{stage_id}].compute.distribution",
            )
            return StageConfig(
                id=stage_id,
                device=raw["device"],
                weights_mb=raw["weights_mb"],
                compute_mode=mode,
                explicit=ExplicitComputeConfig(compute_raw["distribution"]),
                backward=backward,
                memory_placement=memory_placement,
            )

        raise ValueError(
            f"pipeline.stages[{stage_id}].compute.mode must be 'analytical' or 'explicit'"
        )

    def _parse_backward(self, stage_id: int, raw: dict | None) -> BackwardComputeConfig | None:
        if raw is None:
            return None
        if "distribution" not in raw:
            raise ValueError(
                f"pipeline.stages[{stage_id}].backward must define a distribution"
            )
        _require_distribution(
            raw["distribution"], f"pipeline.stages[{stage_id}].backward.distribution"
        )
        return BackwardComputeConfig(distribution=raw["distribution"])

    def _parse_memory_placement(self, stage_id: int, raw: dict) -> MemoryPlacement:
        if not raw:
            return MemoryPlacement()
        kind = raw["kind"]
        if kind == "local":
            return MemoryPlacement(kind="local")
        if kind == "socket":
            if "node" not in raw or "socket" not in raw:
                raise ValueError(
                    f"pipeline.stages[{stage_id}].memory_placement with kind=socket "
                    "requires 'node' and 'socket'"
                )
            return MemoryPlacement(kind="socket", node=raw["node"], socket=str(raw["socket"]))
        if kind == "device":
            if "device" not in raw:
                raise ValueError(
                    f"pipeline.stages[{stage_id}].memory_placement with kind=device "
                    "requires 'device'"
                )
            return MemoryPlacement(kind="device", device=raw["device"])
        raise ValueError(
            f"pipeline.stages[{stage_id}].memory_placement.kind must be local, socket, or device"
        )

    def _parse_hardware(self, raw: dict) -> HardwareConfig:
        devices = [self._parse_device(idx, device_raw) for idx, device_raw in enumerate(raw["devices"])]
        if not devices:
            raise ValueError("hardware.devices must contain at least one device")
        device_ids = [device.id for device in devices]
        if len(set(device_ids)) != len(device_ids):
            raise ValueError(f"hardware.devices contains duplicate ids: {device_ids}")
        interconnect_raw = raw["interconnect"]
        return HardwareConfig(
            devices=devices,
            interconnect=InterconnectConfig(
                same_socket=interconnect_raw.get("same_socket"),
                same_node=interconnect_raw["same_node"],
                cross_node=interconnect_raw["cross_node"],
            ),
        )

    def _parse_device(self, idx: int, raw: dict) -> DeviceSpec:
        preset_keys = [key for key in ("gpu", "cpu") if key in raw]
        if len(preset_keys) != 1:
            raise ValueError(
                f"hardware.devices[{idx}] must define exactly one of 'gpu' or 'cpu'"
            )
        kind = preset_keys[0]
        return DeviceSpec(
            id=raw["id"],
            preset=raw[kind],
            kind=kind,
            node=raw["node"],
            socket=str(raw["socket"]),
        )

    def _parse_optimizer(self, raw: dict) -> OptimizerConfig:
        iteration_barrier = raw.get("iteration_barrier")
        if iteration_barrier is not None:
            _require_distribution(iteration_barrier, "optimizer.iteration_barrier")

        if not raw.get("enabled", False):
            return OptimizerConfig(enabled=False, iteration_barrier=iteration_barrier)

        _require_non_negative(raw.get("gradient_mb", 0.0), "optimizer.gradient_mb")
        _require_positive(raw.get("accumulation_steps", 1), "optimizer.accumulation_steps")
        _require_distribution(raw["update"], "optimizer.update")
        algorithm = raw.get("allreduce", {}).get("algorithm", "naive")
        return OptimizerConfig(
            enabled=True,
            gradient_mb=raw.get("gradient_mb", 0.0),
            accumulation_steps=raw.get("accumulation_steps", 1),
            allreduce_algorithm=AllreduceAlgo(algorithm),
            update_distribution=raw["update"],
            iteration_barrier=iteration_barrier,
        )

    def _parse_failure(self, raw: dict) -> FailureConfig:
        if not raw.get("enabled", False):
            return FailureConfig(enabled=False)
        _require_positive(raw["check_interval_ms"], "failure.check_interval_ms")
        _require_non_negative(
            raw["device_failure_probability"],
            "failure.device_failure_probability",
        )
        _require_non_negative(
            raw["link_failure_probability"],
            "failure.link_failure_probability",
        )
        _require_positive(raw["recovery_time_ms"], "failure.recovery_time_ms")
        return FailureConfig(
            enabled=True,
            check_interval_ms=raw["check_interval_ms"],
            device_failure_probability=raw["device_failure_probability"],
            link_failure_probability=raw["link_failure_probability"],
            recovery_time_ms=raw["recovery_time_ms"],
        )

    def _parse_output(self, raw: dict) -> OutputConfig:
        return OutputConfig(
            timeline=raw.get("timeline"),
            dashboard=raw.get("dashboard"),
            summary_json=raw.get("summary_json"),
            trace_csv=raw.get("trace_csv"),
        )

    def _parse_overrides(self, raw: dict) -> OverridesConfig:
        device_overrides = []
        for idx, device_raw in enumerate(raw.get("devices", [])):
            override = DeviceOverride(
                id=device_raw["id"],
                memory_mb=device_raw.get("memory_mb"),
                flops_tflops=device_raw.get("flops_tflops"),
                memory_bandwidth_gbps=device_raw.get("memory_bandwidth_gbps"),
                launch_overhead_ms=device_raw.get("launch_overhead_ms"),
            )
            if override.memory_mb is not None:
                _require_positive(override.memory_mb, f"overrides.devices[{idx}].memory_mb")
            if override.flops_tflops is not None:
                _require_positive(override.flops_tflops, f"overrides.devices[{idx}].flops_tflops")
            if override.memory_bandwidth_gbps is not None:
                _require_positive(
                    override.memory_bandwidth_gbps,
                    f"overrides.devices[{idx}].memory_bandwidth_gbps",
                )
            if override.launch_overhead_ms is not None:
                _require_non_negative(
                    override.launch_overhead_ms,
                    f"overrides.devices[{idx}].launch_overhead_ms",
                )
            device_overrides.append(override)

        link_overrides = []
        for idx, link_raw in enumerate(raw.get("links", [])):
            _require_positive(link_raw["bandwidth_gbps"], f"overrides.links[{idx}].bandwidth_gbps")
            _require_non_negative(link_raw["latency_us"], f"overrides.links[{idx}].latency_us")
            jitter = link_raw.get("jitter", {"type": "constant", "value": 0.0})
            _require_distribution(jitter, f"overrides.links[{idx}].jitter")
            link_overrides.append(LinkOverride(
                src=link_raw["src"],
                dst=link_raw["dst"],
                bandwidth_gbps=link_raw["bandwidth_gbps"],
                latency_us=link_raw["latency_us"],
                jitter=jitter,
            ))

        return OverridesConfig(devices=device_overrides, links=link_overrides)

    def _validate_stage_devices(self, pipeline: PipelineConfig, hardware: HardwareConfig) -> None:
        known_devices = {device.id for device in hardware.devices}
        for stage in pipeline.stages:
            if stage.device not in known_devices:
                raise ValueError(
                    f"pipeline stage {stage.id} references unknown device {stage.device!r}"
                )
            placement = stage.memory_placement
            if placement.kind == "device" and placement.device not in known_devices:
                raise ValueError(
                    f"pipeline stage {stage.id} memory placement references unknown device "
                    f"{placement.device!r}"
                )


def parse_config(raw: dict) -> AppConfig:
    return ConfigParser().parse(raw)


def validate_config(raw: dict) -> None:
    ConfigParser().parse(raw)
