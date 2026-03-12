"""Summary visualization dashboard."""

import os

import matplotlib.pyplot as plt

from hops.metrics.collector import MetricsCollector


def draw_dashboard(collector: MetricsCollector, output_path: str) -> None:
    """Generate a multi-subplot summary dashboard."""
    if not collector.computes:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. Per-stage utilization bar chart
    ax = axes[0, 0]
    util = collector.per_stage_utilization()
    if util:
        stages = list(util.keys())
        values = [util[s] for s in stages]
        ax.bar([f"Stage {s}" for s in stages], values, color="#4C9BE8")
        ax.set_ylabel("Utilization")
        ax.set_title("Per-Stage Utilization")
        ax.set_ylim(0, 1)

    # 2. E2E latency histogram
    ax = axes[0, 1]
    latencies = collector.e2e_latencies()
    if latencies:
        ax.hist(latencies, bins=max(5, len(latencies) // 3), color="#E8804C",
                edgecolor="white")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title("End-to-End Latency Distribution")

    # 3. Bubble ratio pie chart
    ax = axes[1, 0]
    bubble = collector.bubble_ratio()
    ax.pie([1 - bubble, bubble], labels=["Compute", "Bubble"],
           colors=["#4C9BE8", "#CCCCCC"], autopct="%.1f%%", startangle=90)
    ax.set_title("Bubble Ratio")

    # 4. Communication vs compute time
    ax = axes[1, 1]
    total_compute = collector.total_compute_time()
    total_transfer = collector.total_transfer_time()
    if total_compute + total_transfer > 0:
        ax.bar(["Compute", "Communication"], [total_compute, total_transfer],
               color=["#4C9BE8", "#E8804C"])
        ax.set_ylabel("Total Time (ms)")
        ax.set_title("Compute vs Communication")

    fig.suptitle("HOPS Simulation Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Dashboard saved to {output_path}")
