"""Throughput, bubble ratio, and utilization reporting."""

import numpy as np

from hops.metrics.collector import MetricsCollector


class Reporter:
    """Prints a summary of simulation metrics."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def print_summary(self) -> None:
        c = self.collector
        print("\n" + "=" * 60)
        print("HOPS Simulation Report")
        print("=" * 60)

        print(f"\nMicro-batches completed: {c.completed_microbatches}")
        print(f"Throughput: {c.throughput():.4f} micro-batches/ms")

        latencies = c.e2e_latencies()
        if latencies:
            arr = np.array(latencies)
            print(f"\nEnd-to-end latency:")
            print(f"  p50: {np.percentile(arr, 50):.2f} ms")
            print(f"  p99: {np.percentile(arr, 99):.2f} ms")
            print(f"  mean: {np.mean(arr):.2f} ms")

        print(f"\nBubble ratio: {c.bubble_ratio():.2%}")

        util = c.per_stage_utilization()
        if util:
            print("\nPer-stage utilization:")
            for stage_id, u in util.items():
                print(f"  Stage {stage_id}: {u:.2%}")

        total_compute = c.total_compute_time()
        total_transfer = c.total_transfer_time()
        if total_compute > 0 and c.transfers:
            print(f"\nCommunication overhead: {total_transfer / total_compute:.2%} of compute")

        if c.failures:
            print(f"\nFailures: {len(c.failures)}")
            total_downtime = sum(f.recovery_time for f in c.failures)
            print(f"Total downtime: {total_downtime:.2f} ms")

        print("\n" + "=" * 60)
