"""Runtime statistics accumulation."""

from dataclasses import dataclass

from hops.core.types import Phase


@dataclass
class ComputeRecord:
    stage_id: int
    microbatch_id: int
    phase: Phase
    device_id: str
    start_time: float
    end_time: float


@dataclass
class TransferRecord:
    microbatch_id: int
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
    """Collects simulation events and computes derived metrics."""

    def __init__(self):
        self.computes: list[ComputeRecord] = []
        self.transfers: list[TransferRecord] = []
        self.failures: list[FailureRecord] = []
        self.in_flight: list[InFlightRecord] = []
        self._mb_start_times: dict[int, float] = {}
        self._mb_completion_times: dict[int, float] = {}

    def record_compute(self, stage_id: int, microbatch_id: int, phase: Phase,
                       device_id: str, start_time: float, end_time: float) -> None:
        self.computes.append(ComputeRecord(
            stage_id, microbatch_id, phase, device_id, start_time, end_time))
        # Track per-microbatch start/end for e2e latency
        if microbatch_id not in self._mb_start_times:
            self._mb_start_times[microbatch_id] = start_time
        else:
            self._mb_start_times[microbatch_id] = min(
                self._mb_start_times[microbatch_id], start_time)

    def record_transfer(self, microbatch_id: int, phase: Phase,
                        src: str, dst: str, start: float, end: float) -> None:
        self.transfers.append(TransferRecord(
            microbatch_id, phase, src, dst, start, end))

    def record_failure(self, target_id: str, time: float, recovery_time: float) -> None:
        self.failures.append(FailureRecord(target_id, time, recovery_time))

    def record_in_flight(self, stage_id: int, time: float, count: int) -> None:
        self.in_flight.append(InFlightRecord(stage_id, time, count))

    def record_microbatch_completion(self, microbatch_id: int, time: float) -> None:
        self._mb_completion_times[microbatch_id] = time

    @property
    def completed_microbatches(self) -> int:
        return len(self._mb_completion_times)

    def e2e_latencies(self) -> list[float]:
        """Per-microbatch end-to-end latency."""
        return [self._mb_completion_times[mb] - self._mb_start_times[mb]
                for mb in sorted(self._mb_completion_times)
                if mb in self._mb_start_times]

    def throughput(self) -> float:
        """Micro-batches per time unit."""
        if not self._mb_completion_times:
            return 0.0
        total_time = max(self._mb_completion_times.values()) - min(
            self._mb_start_times[mb] for mb in self._mb_completion_times)
        if total_time <= 0:
            return 0.0
        return len(self._mb_completion_times) / total_time

    def total_compute_time(self) -> float:
        return sum(r.end_time - r.start_time for r in self.computes)

    def total_transfer_time(self) -> float:
        return sum(t.end_time - t.start_time for t in self.transfers)

    def makespan(self) -> float:
        if not self.computes:
            return 0.0
        return max(r.end_time for r in self.computes) - min(r.start_time for r in self.computes)

    def stage_occupancy_intervals(self) -> dict[int, list[tuple[float, float]]]:
        intervals: dict[int, list[tuple[float, float]]] = {}
        for record in sorted(self.computes, key=lambda r: (r.stage_id, r.start_time, r.end_time)):
            stage_intervals = intervals.setdefault(record.stage_id, [])
            if not stage_intervals or record.start_time > stage_intervals[-1][1]:
                stage_intervals.append((record.start_time, record.end_time))
                continue
            start, end = stage_intervals[-1]
            stage_intervals[-1] = (start, max(end, record.end_time))
        return intervals

    def stage_idle_intervals(self) -> dict[int, list[tuple[float, float]]]:
        occupancy = self.stage_occupancy_intervals()
        if not occupancy:
            return {}

        trace_start = min(start for intervals in occupancy.values() for start, _ in intervals)
        trace_end = max(end for intervals in occupancy.values() for _, end in intervals)
        idle: dict[int, list[tuple[float, float]]] = {}

        for stage_id, intervals in occupancy.items():
            stage_idle: list[tuple[float, float]] = []
            cursor = trace_start
            for start, end in intervals:
                if start > cursor:
                    stage_idle.append((cursor, start))
                cursor = max(cursor, end)
            if cursor < trace_end:
                stage_idle.append((cursor, trace_end))
            idle[stage_id] = stage_idle
        return idle

    def per_stage_utilization(self) -> dict[int, float]:
        """Fraction of total time each stage spent computing."""
        occupancy = self.stage_occupancy_intervals()
        if not occupancy:
            return {}
        total_time = self.makespan()
        if total_time <= 0:
            return {}
        return {
            stage_id: sum(end - start for start, end in intervals) / total_time
            for stage_id, intervals in sorted(occupancy.items())
        }

    def bubble_ratio(self) -> float:
        """Fraction of total device-time that is idle (pipeline bubbles)."""
        idle = self.stage_idle_intervals()
        if not idle:
            return 0.0
        total_time = self.makespan()
        if total_time <= 0:
            return 0.0
        total_idle = sum(end - start for intervals in idle.values() for start, end in intervals)
        return total_idle / (total_time * len(idle))

    def peak_in_flight_per_stage(self) -> dict[int, int]:
        peaks: dict[int, int] = {}
        for record in self.in_flight:
            peaks[record.stage_id] = max(peaks.get(record.stage_id, 0), record.count)
        return peaks

    def reset(self) -> None:
        """Clear all records for a fresh simulation."""
        self.computes.clear()
        self.transfers.clear()
        self.failures.clear()
        self.in_flight.clear()
        self._mb_start_times.clear()
        self._mb_completion_times.clear()
