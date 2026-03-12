"""Runtime statistics accumulation."""

from dataclasses import dataclass, field

import numpy as np

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


class MetricsCollector:
    """Collects simulation events and computes derived metrics."""

    def __init__(self):
        self.computes: list[ComputeRecord] = []
        self.transfers: list[TransferRecord] = []
        self.failures: list[FailureRecord] = []
        self._batch_completions: list[float] = []
        self._mb_start_times: dict[int, float] = {}
        self._mb_end_times: dict[int, float] = {}

    def record_compute(self, stage_id: int, microbatch_id: int, phase: Phase,
                       device_id: str, start_time: float, end_time: float) -> None:
        self.computes.append(ComputeRecord(
            stage_id, microbatch_id, phase, device_id, start_time, end_time))
        # Track per-microbatch start/end for e2e latency
        if microbatch_id not in self._mb_start_times:
            self._mb_start_times[microbatch_id] = start_time
        self._mb_start_times[microbatch_id] = min(
            self._mb_start_times[microbatch_id], start_time)
        self._mb_end_times[microbatch_id] = max(
            self._mb_end_times.get(microbatch_id, 0.0), end_time)

    def record_transfer(self, microbatch_id: int, phase: Phase,
                        src: str, dst: str, start: float, end: float) -> None:
        self.transfers.append(TransferRecord(
            microbatch_id, phase, src, dst, start, end))

    def record_failure(self, target_id: str, time: float, recovery_time: float) -> None:
        self.failures.append(FailureRecord(target_id, time, recovery_time))

    def record_batch_completion(self, time: float) -> None:
        self._batch_completions.append(time)

    def e2e_latencies(self) -> list[float]:
        """Per-microbatch end-to-end latency."""
        return [self._mb_end_times[mb] - self._mb_start_times[mb]
                for mb in sorted(self._mb_start_times)
                if mb in self._mb_end_times]

    def throughput(self) -> float:
        """Micro-batches per time unit."""
        if not self._mb_end_times:
            return 0.0
        total_time = max(self._mb_end_times.values()) - min(self._mb_start_times.values())
        if total_time <= 0:
            return 0.0
        return len(self._mb_end_times) / total_time

    def per_stage_utilization(self) -> dict[int, float]:
        """Fraction of total time each stage spent computing."""
        if not self.computes:
            return {}
        total_time = max(r.end_time for r in self.computes) - min(r.start_time for r in self.computes)
        if total_time <= 0:
            return {}
        by_stage: dict[int, float] = {}
        for r in self.computes:
            by_stage[r.stage_id] = by_stage.get(r.stage_id, 0.0) + (r.end_time - r.start_time)
        return {s: t / total_time for s, t in sorted(by_stage.items())}

    def bubble_ratio(self) -> float:
        """Fraction of total device-time that is idle (pipeline bubbles)."""
        if not self.computes:
            return 0.0
        utilization = self.per_stage_utilization()
        if not utilization:
            return 0.0
        avg_util = np.mean(list(utilization.values()))
        return 1.0 - avg_util

    def reset(self) -> None:
        """Clear all records for a fresh simulation."""
        self.computes.clear()
        self.transfers.clear()
        self.failures.clear()
        self._batch_completions.clear()
        self._mb_start_times.clear()
        self._mb_end_times.clear()
