"""Raw runtime event collection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComputeRecord:
    stage_id: int
    microbatch_id: int | None
    phase: Phase
    device_id: str
    start_time: float
    end_time: float


@dataclass
class TransferRecord:
    microbatch_id: int | None
    phase: Phase
    src_device: str
    dst_device: str
    start_time: float
    end_time: float


@dataclass
class FailureRecord:
    target_id: str
    time: float
    recovery_time: float


@dataclass
class InFlightRecord:
    stage_id: int
    time: float
    count: int


class MetricsCollector:
    """Append-only store for simulation events."""

    def __init__(self):
        self.computes: list[ComputeRecord] = []
        self.transfers: list[TransferRecord] = []
        self.failures: list[FailureRecord] = []
        self.in_flight: list[InFlightRecord] = []
        self._mb_start_times: dict[int, float] = {}
        self._mb_completion_times: dict[int, float] = {}
        self.peak_memory_per_device: dict[str, float] = {}

    def record_compute(self, stage_id: int, microbatch_id: int | None, phase: Phase,
                       device_id: str, start_time: float, end_time: float) -> None:
        self.computes.append(ComputeRecord(
            stage_id, microbatch_id, phase, device_id, start_time, end_time))
        # Track per-microbatch start/end for e2e latency (skip optimizer records)
        if microbatch_id is not None:
            if microbatch_id not in self._mb_start_times:
                self._mb_start_times[microbatch_id] = start_time
            else:
                self._mb_start_times[microbatch_id] = min(
                    self._mb_start_times[microbatch_id], start_time)

    def record_transfer(self, microbatch_id: int | None, phase: Phase,
                        src: str, dst: str, start: float, end: float) -> None:
        self.transfers.append(TransferRecord(
            microbatch_id, phase, src, dst, start, end))

    def record_failure(self, target_id: str, time: float, recovery_time: float) -> None:
        self.failures.append(FailureRecord(target_id, time, recovery_time))

    def record_in_flight(self, stage_id: int, time: float, count: int) -> None:
        self.in_flight.append(InFlightRecord(stage_id, time, count))

    def record_microbatch_completion(self, microbatch_id: int, time: float) -> None:
        self._mb_completion_times[microbatch_id] = time

    def record_peak_memory(self, device_id: str, peak_mb: float) -> None:
        self.peak_memory_per_device[device_id] = max(
            self.peak_memory_per_device.get(device_id, 0.0), peak_mb)

    @property
    def completed_microbatches(self) -> int:
        return len(self._mb_completion_times)

    @property
    def microbatch_start_times(self) -> dict[int, float]:
        return self._mb_start_times

    @property
    def microbatch_completion_times(self) -> dict[int, float]:
        return self._mb_completion_times

    def _analyzer(self):
        from hops.metrics.analyzer import MetricsAnalyzer

        return MetricsAnalyzer(self)

    def trace_duration(self) -> float:
        return self._analyzer().trace_duration()

    def e2e_latencies(self) -> list[float]:
        return self._analyzer().e2e_latencies()

    def throughput(self) -> float:
        return self._analyzer().throughput()

    def total_compute_time(self) -> float:
        return self._analyzer().total_compute_time()

    def total_transfer_time(self) -> float:
        return self._analyzer().total_transfer_time()

    def makespan(self) -> float:
        return self._analyzer().makespan()

    def stage_occupancy_intervals(self) -> dict[int, list[tuple[float, float]]]:
        return self._analyzer().stage_occupancy_intervals()

    def device_occupancy_intervals(self) -> dict[str, list[tuple[float, float]]]:
        return self._analyzer().device_occupancy_intervals()

    def link_occupancy_intervals(self) -> dict[str, list[tuple[float, float]]]:
        return self._analyzer().link_occupancy_intervals()

    def stage_idle_intervals(self) -> dict[int, list[tuple[float, float]]]:
        return self._analyzer().stage_idle_intervals()

    def per_stage_utilization(self) -> dict[int, float]:
        return self._analyzer().per_stage_utilization()

    def per_device_utilization(self) -> dict[str, float]:
        return self._analyzer().per_device_utilization()

    def per_link_transfer_utilization(self) -> dict[str, float]:
        return self._analyzer().per_link_transfer_utilization()

    def transfer_contention_stats(self) -> dict[str, object]:
        return self._analyzer().transfer_contention_stats()

    def summary(self) -> dict[str, object]:
        return self._analyzer().summary().to_dict()

    def export_trace_csv(self, output_path: str) -> None:
        from hops.metrics.exporter import TraceExporter

        TraceExporter(self).write_csv(output_path)

    def bubble_ratio(self) -> float:
        return self._analyzer().bubble_ratio()

    def peak_in_flight_per_stage(self) -> dict[int, int]:
        return self._analyzer().peak_in_flight_per_stage()

    def reset(self) -> None:
        """Clear all records for a fresh simulation."""
        self.computes.clear()
        self.transfers.clear()
        self.failures.clear()
        self.in_flight.clear()
        self._mb_start_times.clear()
        self._mb_completion_times.clear()
        self.peak_memory_per_device.clear()
