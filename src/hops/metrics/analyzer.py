"""Derived analytics for recorded simulation metrics."""

from __future__ import annotations

import numpy as np

from hops.core.types import Phase
from hops.metrics.collector import MetricsCollector, TransferRecord
from hops.metrics.summary import (
    ContentionSummary,
    FailureSummary,
    LatencySummary,
    MemorySummary,
    OptimizerSummary,
    SimulationSummary,
    ThroughputSummary,
    TimeSummary,
    UtilizationSummary,
)


class MetricsAnalyzer:
    """Compute derived metrics from raw collector records."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    @staticmethod
    def _merge_intervals(
        records: list[tuple[str | int, float, float]],
    ) -> dict[str | int, list[tuple[float, float]]]:
        intervals: dict[str | int, list[tuple[float, float]]] = {}
        for key, start_time, end_time in sorted(records, key=lambda row: (row[0], row[1], row[2])):
            merged = intervals.setdefault(key, [])
            if not merged or start_time > merged[-1][1]:
                merged.append((start_time, end_time))
                continue
            start, end = merged[-1]
            merged[-1] = (start, max(end, end_time))
        return intervals

    def trace_bounds(self) -> tuple[float, float]:
        starts = [record.start_time for record in self.collector.computes]
        starts.extend(record.start_time for record in self.collector.transfers)
        ends = [record.end_time for record in self.collector.computes]
        ends.extend(record.end_time for record in self.collector.transfers)
        if not starts or not ends:
            return 0.0, 0.0
        return min(starts), max(ends)

    def trace_duration(self) -> float:
        start, end = self.trace_bounds()
        return max(0.0, end - start)

    def e2e_latencies(self) -> list[float]:
        return [
            self.collector.microbatch_completion_times[mb] - self.collector.microbatch_start_times[mb]
            for mb in sorted(self.collector.microbatch_completion_times)
            if mb in self.collector.microbatch_start_times
        ]

    def throughput(self) -> float:
        if not self.collector.microbatch_completion_times:
            return 0.0
        total_time = max(self.collector.microbatch_completion_times.values()) - min(
            self.collector.microbatch_start_times[mb]
            for mb in self.collector.microbatch_completion_times
        )
        if total_time <= 0:
            return 0.0
        return len(self.collector.microbatch_completion_times) / total_time

    def total_compute_time(self) -> float:
        return sum(record.end_time - record.start_time for record in self.collector.computes)

    def total_transfer_time(self) -> float:
        return sum(record.end_time - record.start_time for record in self.collector.transfers)

    def makespan(self) -> float:
        if not self.collector.computes:
            return 0.0
        return max(record.end_time for record in self.collector.computes) - min(
            record.start_time for record in self.collector.computes
        )

    def stage_occupancy_intervals(self) -> dict[int, list[tuple[float, float]]]:
        return self._merge_intervals([
            (record.stage_id, record.start_time, record.end_time)
            for record in self.collector.computes
        ])

    def device_occupancy_intervals(self) -> dict[str, list[tuple[float, float]]]:
        return self._merge_intervals([
            (record.device_id, record.start_time, record.end_time)
            for record in self.collector.computes
        ])

    def link_occupancy_intervals(self) -> dict[str, list[tuple[float, float]]]:
        return self._merge_intervals([
            (f"{record.src_device}->{record.dst_device}", record.start_time, record.end_time)
            for record in self.collector.transfers
        ])

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

    def bubble_ratio(self) -> float:
        idle = self.stage_idle_intervals()
        if not idle:
            return 0.0
        total_time = self.makespan()
        if total_time <= 0:
            return 0.0
        total_idle = sum(end - start for intervals in idle.values() for start, end in intervals)
        return total_idle / (total_time * len(idle))

    def per_stage_utilization(self) -> dict[int, float]:
        occupancy = self.stage_occupancy_intervals()
        total_time = self.makespan()
        if not occupancy or total_time <= 0:
            return {}
        return {
            stage_id: sum(end - start for start, end in intervals) / total_time
            for stage_id, intervals in sorted(occupancy.items())
        }

    def per_device_utilization(self) -> dict[str, float]:
        occupancy = self.device_occupancy_intervals()
        total_time = self.trace_duration()
        if not occupancy or total_time <= 0:
            return {}
        return {
            device_id: sum(end - start for start, end in intervals) / total_time
            for device_id, intervals in sorted(occupancy.items())
        }

    def per_link_transfer_utilization(self) -> dict[str, float]:
        occupancy = self.link_occupancy_intervals()
        total_time = self.trace_duration()
        if not occupancy or total_time <= 0:
            return {}
        return {
            link_id: sum(end - start for start, end in intervals) / total_time
            for link_id, intervals in sorted(occupancy.items())
        }

    def transfer_contention_stats(self) -> dict[str, object]:
        per_link: dict[str, list[TransferRecord]] = {}
        for record in self.collector.transfers:
            per_link.setdefault(f"{record.src_device}->{record.dst_device}", []).append(record)

        per_link_summary: dict[str, dict[str, float]] = {}
        total_transfers = 0
        total_contended = 0
        global_peak = 0

        for link_id, records in sorted(per_link.items()):
            ordered = sorted(records, key=lambda record: (record.start_time, record.end_time))
            events: list[tuple[float, int]] = []
            for record in ordered:
                events.append((record.start_time, 1))
                events.append((record.end_time, -1))
            events.sort(key=lambda item: (item[0], item[1]))

            active = 0
            peak = 0
            overlaps_previous = [False] * len(ordered)
            overlaps_future = [False] * len(ordered)

            max_end_so_far = float("-inf")
            for idx, record in enumerate(ordered):
                if record.start_time < max_end_so_far:
                    overlaps_previous[idx] = True
                max_end_so_far = max(max_end_so_far, record.end_time)

            min_future_start = float("inf")
            for idx in range(len(ordered) - 1, -1, -1):
                record = ordered[idx]
                if record.end_time > min_future_start:
                    overlaps_future[idx] = True
                min_future_start = min(min_future_start, record.start_time)

            contended = sum(
                1 for idx in range(len(ordered)) if overlaps_previous[idx] or overlaps_future[idx]
            )

            for _, delta in events:
                active += delta
                peak = max(peak, active)

            count = len(ordered)
            total_transfers += count
            total_contended += contended
            global_peak = max(global_peak, peak)
            per_link_summary[link_id] = {
                "transfer_count": float(count),
                "peak_concurrency": float(peak),
                "contended_transfer_fraction": contended / count if count else 0.0,
            }

        return {
            "global_peak_concurrency": float(global_peak),
            "global_contended_transfer_fraction": (
                total_contended / total_transfers if total_transfers else 0.0
            ),
            "per_link": per_link_summary,
        }

    def peak_in_flight_per_stage(self) -> dict[int, int]:
        peaks: dict[int, int] = {}
        for record in self.collector.in_flight:
            peaks[record.stage_id] = max(peaks.get(record.stage_id, 0), record.count)
        return peaks

    def optimizer_summary(self) -> OptimizerSummary:
        optimizer_compute = sum(
            record.end_time - record.start_time
            for record in self.collector.computes
            if record.phase == Phase.OPTIMIZER
        )
        optimizer_transfer = sum(
            record.end_time - record.start_time
            for record in self.collector.transfers
            if record.phase == Phase.OPTIMIZER
        )
        return OptimizerSummary(
            allreduce_time_ms=optimizer_transfer,
            weight_update_time_ms=optimizer_compute,
        )

    def failure_summary(self) -> FailureSummary:
        total_downtime = sum(record.recovery_time for record in self.collector.failures)
        return FailureSummary(
            count=len(self.collector.failures),
            total_downtime_ms=total_downtime,
            lost_work_ms=None,
        )

    def latency_summary(self) -> LatencySummary:
        latencies = self.e2e_latencies()
        if not latencies:
            return LatencySummary()
        arr = np.array(latencies)
        return LatencySummary(
            p50_ms=float(np.percentile(arr, 50)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
        )

    def summary(self) -> SimulationSummary:
        throughput = self.throughput()
        total_compute = self.total_compute_time()
        total_transfer = self.total_transfer_time()
        contention = self.transfer_contention_stats()
        return SimulationSummary(
            completed_microbatches=self.collector.completed_microbatches,
            throughput=ThroughputSummary(per_ms=throughput, per_s=throughput * 1000.0),
            latency_ms=self.latency_summary(),
            bubble_ratio=self.bubble_ratio(),
            time_ms=TimeSummary(
                trace_duration_ms=self.trace_duration(),
                makespan_ms=self.makespan(),
                compute_ms=total_compute,
                transfer_ms=total_transfer,
                communication_overhead_ratio=(total_transfer / total_compute if total_compute > 0 else 0.0),
            ),
            utilization=UtilizationSummary(
                per_stage=self.per_stage_utilization(),
                per_device=self.per_device_utilization(),
                per_link=self.per_link_transfer_utilization(),
            ),
            optimizer=self.optimizer_summary(),
            failures=self.failure_summary(),
            memory=MemorySummary(
                peak_per_device_mb=dict(sorted(self.collector.peak_memory_per_device.items()))
            ),
            contention=ContentionSummary(
                global_peak_concurrency=contention["global_peak_concurrency"],
                global_contended_transfer_fraction=contention["global_contended_transfer_fraction"],
                per_link=contention["per_link"],
            ),
            peak_in_flight_per_stage=self.peak_in_flight_per_stage(),
        )
