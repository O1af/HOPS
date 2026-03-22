"""Summary visualization dashboard."""

import os

import matplotlib.pyplot as plt

from hops.metrics.summary import SimulationSummary


def draw_dashboard(summary: SimulationSummary, latencies: list[float], output_path: str) -> None:
    """Generate a multi-subplot summary dashboard."""
    if summary.completed_microbatches == 0:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    # 1. Per-stage utilization bar chart
    ax = axes[0, 0]
    if summary.utilization.per_stage:
        stages = list(summary.utilization.per_stage.keys())
        values = [summary.utilization.per_stage[s] for s in stages]
        ax.bar([f"Stage {s}" for s in stages], values, color="#4C9BE8")
        ax.set_ylabel("Utilization")
        ax.set_title("Per-Stage Utilization")
        ax.set_ylim(0, 1)

    # 2. Per-device utilization bar chart
    ax = axes[0, 1]
    if summary.utilization.per_device:
        devices = list(summary.utilization.per_device.keys())
        values = [summary.utilization.per_device[d] for d in devices]
        ax.bar(devices, values, color="#6BC86B")
        ax.set_ylabel("Utilization")
        ax.set_title("Per-Device Utilization")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)

    # 3. E2E latency histogram
    ax = axes[1, 0]
    if latencies:
        ax.hist(latencies, bins=max(5, len(latencies) // 3), color="#E8804C",
                edgecolor="white")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title("End-to-End Latency Distribution")

    # 4. Bubble ratio pie chart
    ax = axes[1, 1]
    bubble = summary.bubble_ratio
    ax.pie([1 - bubble, bubble], labels=["Compute", "Bubble"],
           colors=["#4C9BE8", "#CCCCCC"], autopct="%.1f%%", startangle=90)
    ax.set_title("Bubble Ratio")

    # 5. Communication vs compute time
    ax = axes[2, 0]
    total_compute = summary.time_ms.compute_ms
    total_transfer = summary.time_ms.transfer_ms
    if total_compute + total_transfer > 0:
        ax.bar(["Compute", "Communication"], [total_compute, total_transfer],
               color=["#4C9BE8", "#E8804C"])
        ax.set_ylabel("Total Time (ms)")
        ax.set_title("Compute vs Communication")

    # 6. Link transfer utilization
    ax = axes[2, 1]
    if summary.utilization.per_link:
        links = list(summary.utilization.per_link.keys())
        values = [summary.utilization.per_link[link] for link in links]
        ax.bar(links, values, color="#D4A843")
        ax.set_ylabel("Utilization")
        ax.set_ylim(0, 1)
        ax.set_title("Per-Link Transfer Utilization")
        ax.tick_params(axis="x", rotation=20)
        for idx, link in enumerate(links):
            peak = summary.contention.per_link.get(link, {}).get("peak_concurrency", 0.0)
            ax.text(idx, values[idx], f"x{peak:.0f}", ha="center", va="bottom", fontsize=8)

    # 7. Peak memory per device
    ax = axes[3, 0]
    if summary.memory.peak_per_device_mb:
        devices = list(summary.memory.peak_per_device_mb.keys())
        values = [summary.memory.peak_per_device_mb[d] for d in devices]
        ax.bar(devices, values, color="#B07BEC")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Peak Memory Per Device")
        ax.tick_params(axis="x", rotation=20)

    # 8. Text summary panel
    ax = axes[3, 1]
    ax.axis("off")
    lines = [
        f"Throughput: {summary.throughput.per_s:.2f} micro-batches/s",
        f"Bubble ratio: {summary.bubble_ratio:.1%}",
        f"Comm overhead: {summary.time_ms.communication_overhead_ratio:.1%}",
        f"Peak link concurrency: {summary.contention.global_peak_concurrency:.0f}",
        f"All-reduce time: {summary.optimizer.allreduce_time_ms:.2f} ms",
        f"Failure downtime: {summary.failures.total_downtime_ms:.2f} ms",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10)
    ax.set_title("Key Metrics")

    fig.suptitle("HOPS Simulation Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Dashboard saved to {output_path}")
