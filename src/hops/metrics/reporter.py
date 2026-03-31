"""Throughput, bubble ratio, and utilization reporting."""

import json
from pathlib import Path

from hops.metrics.analyzer import MetricsAnalyzer
from hops.metrics.collector import MetricsCollector
from hops.metrics.summary import SimulationSummary


class Reporter:
    """Prints a summary of simulation metrics."""

    def __init__(self, metrics: MetricsCollector | MetricsAnalyzer):
        self._analyzer = metrics if isinstance(metrics, MetricsAnalyzer) else metrics.analyzer

    @property
    def analyzer(self) -> MetricsAnalyzer:
        return self._analyzer

    def summary(self) -> dict[str, object]:
        return self.summary_model().to_dict()

    def summary_model(self) -> SimulationSummary:
        return self.analyzer.summary()

    def write_summary_json(self, output_path: str,
                           summary: SimulationSummary | None = None) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (summary or self.summary_model()).to_dict()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def print_summary(self, summary: SimulationSummary | None = None) -> None:
        summary = summary or self.summary_model()
        print("\n" + "=" * 60)
        print("HOPS Simulation Report")
        print("=" * 60)

        print(f"\nMicro-batches completed: {summary.completed_microbatches}")
        print(f"Throughput: {summary.throughput.per_ms:.4f} micro-batches/ms")
        print(f"            {summary.throughput.per_s:.2f} micro-batches/s")

        if summary.latency_ms.mean_ms is not None:
            print("\nEnd-to-end latency:")
            print(f"  p50: {summary.latency_ms.p50_ms:.2f} ms")
            print(f"  p99: {summary.latency_ms.p99_ms:.2f} ms")
            print(f"  mean: {summary.latency_ms.mean_ms:.2f} ms")

        print(f"\nBubble ratio: {summary.bubble_ratio:.2%}")

        if summary.utilization.per_stage:
            print("\nPer-stage utilization:")
            for stage_id, u in summary.utilization.per_stage.items():
                print(f"  Stage {stage_id}: {u:.2%}")

        if summary.utilization.per_device:
            print("\nPer-device utilization:")
            for device_id, u in summary.utilization.per_device.items():
                print(f"  {device_id}: {u:.2%}")

        if summary.time_ms.compute_ms > 0:
            print(
                "\nCommunication overhead: "
                f"{summary.time_ms.communication_overhead_ratio:.2%} of compute"
            )

        if summary.utilization.per_link:
            print("\nPer-link transfer utilization:")
            for link_id, u in summary.utilization.per_link.items():
                print(f"  {link_id}: {u:.2%}")

        if summary.contention.per_link:
            print("\nTransfer contention:")
            print(f"  Global peak concurrency: {summary.contention.global_peak_concurrency:.0f}")
            print(
                "  Contended transfer fraction: "
                f"{summary.contention.global_contended_transfer_fraction:.2%}"
            )

        if summary.optimizer.allreduce_time_ms > 0 or summary.optimizer.weight_update_time_ms > 0:
            print(f"\nOptimizer step:")
            print(f"  All-reduce time: {summary.optimizer.allreduce_time_ms:.2f} ms")
            print(f"  Weight update time: {summary.optimizer.weight_update_time_ms:.2f} ms")

        if summary.memory.peak_per_device_mb:
            print(f"\nPeak memory per device:")
            for device_id, peak in sorted(summary.memory.peak_per_device_mb.items()):
                print(f"  {device_id}: {peak:.1f} MB")

        if summary.failures.count:
            print(f"\nFailures: {summary.failures.count}")
            print(f"Total downtime: {summary.failures.total_downtime_ms:.2f} ms")
            if summary.failures.lost_work_ms is None:
                print("Lost work: n/a")
            else:
                print(f"Lost work: {summary.failures.lost_work_ms:.2f} ms")

        print("\n" + "=" * 60)
