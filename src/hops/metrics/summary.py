"""Typed simulation summary models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class ThroughputSummary:
    per_ms: float = 0.0
    per_s: float = 0.0


@dataclass(frozen=True)
class LatencySummary:
    p50_ms: float | None = None
    p99_ms: float | None = None
    mean_ms: float | None = None


@dataclass(frozen=True)
class TimeSummary:
    trace_duration_ms: float = 0.0
    makespan_ms: float = 0.0
    compute_ms: float = 0.0
    transfer_ms: float = 0.0
    communication_overhead_ratio: float = 0.0


@dataclass(frozen=True)
class UtilizationSummary:
    per_stage: dict[int, float] = field(default_factory=dict)
    per_device: dict[str, float] = field(default_factory=dict)
    per_link: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerSummary:
    allreduce_time_ms: float = 0.0
    weight_update_time_ms: float = 0.0


@dataclass(frozen=True)
class FailureSummary:
    count: int = 0
    total_downtime_ms: float = 0.0
    lost_work_ms: float | None = None


@dataclass(frozen=True)
class MemorySummary:
    peak_per_device_mb: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ContentionSummary:
    global_peak_concurrency: float = 0.0
    global_contended_transfer_fraction: float = 0.0
    per_link: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseBreakdownEntry:
    stage: int
    phase: str
    total_ms: float
    count: int


@dataclass(frozen=True)
class SimulationSummary:
    completed_microbatches: int
    throughput: ThroughputSummary
    latency_ms: LatencySummary
    bubble_ratio: float
    time_ms: TimeSummary
    utilization: UtilizationSummary
    optimizer: OptimizerSummary
    failures: FailureSummary
    memory: MemorySummary
    contention: ContentionSummary
    peak_in_flight_per_stage: dict[int, int] = field(default_factory=dict)
    phase_breakdown: list[PhaseBreakdownEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
