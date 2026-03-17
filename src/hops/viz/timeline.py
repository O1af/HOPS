"""Gantt-style pipeline timeline plots."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from hops.core.types import Phase
from hops.metrics.collector import MetricsCollector


COLORS = {
    Phase.FORWARD: "#4C9BE8",
    Phase.BACKWARD: "#E8804C",
    Phase.BACKWARD_B: "#E8804C",   # activation gradient — same hue as backward
    Phase.BACKWARD_W: "#D4A843",   # weight gradient — distinct gold
    Phase.OPTIMIZER: "#6BC86B",
}


def draw_timeline(collector: MetricsCollector, output_path: str) -> None:
    """Draw a Gantt chart of compute events per device."""
    if not collector.computes:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Collect unique devices in stage order
    devices = []
    seen = set()
    for r in sorted(collector.computes, key=lambda r: r.stage_id):
        if r.device_id not in seen:
            devices.append(r.device_id)
            seen.add(r.device_id)

    device_y = {d: i for i, d in enumerate(devices)}

    fig, ax = plt.subplots(figsize=(14, max(3, len(devices) * 0.8)))

    for r in collector.computes:
        y = device_y[r.device_id]
        width = r.end_time - r.start_time
        color = COLORS[r.phase]
        ax.barh(y, width, left=r.start_time, height=0.6, color=color,
                edgecolor="white", linewidth=0.5)
        # Label with microbatch id (skip for optimizer records)
        if width > 0.5 and r.microbatch_id is not None:
            ax.text(r.start_time + width / 2, y, str(r.microbatch_id),
                    ha="center", va="center", fontsize=7, color="white")

    # Draw failure markers
    for f in collector.failures:
        if f.target_id in device_y:
            ax.plot(f.time, device_y[f.target_id], "rx", markersize=10, markeredgewidth=2)

    ax.set_yticks(range(len(devices)))
    ax.set_yticklabels(devices)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Pipeline Timeline")

    # Build legend from phases actually present
    present_phases = {r.phase for r in collector.computes}
    legend_items = [
        (Phase.FORWARD, "Forward"),
        (Phase.BACKWARD, "Backward"),
        (Phase.BACKWARD_B, "Backward-B"),
        (Phase.BACKWARD_W, "Backward-W"),
        (Phase.OPTIMIZER, "Optimizer"),
    ]
    ax.legend(
        handles=[mpatches.Patch(color=COLORS[p], label=label)
                 for p, label in legend_items if p in present_phases],
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Timeline saved to {output_path}")
