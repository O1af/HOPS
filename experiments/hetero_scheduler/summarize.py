"""Summarize hetero scheduler results from final_results.json."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    data = json.loads((here / "final_results.json").read_text())

    schedulers = sorted(next(iter(data.values())).keys())
    baseline = "zero_bubble"

    print(f"{'config':<40s} ", end="")
    for sched in schedulers:
        print(f"{sched[:18]:>18s} ", end="")
    print()
    print("-" * (40 + len(schedulers) * 19))

    total_delta = {s: 0.0 for s in schedulers}
    wins = {s: 0 for s in schedulers}
    losses = {s: 0 for s in schedulers}
    ties = {s: 0 for s in schedulers}

    for cfg, scheds in data.items():
        base_ms = scheds[baseline]["avg"]["makespan_ms"]
        print(f"{cfg:<40s} ", end="")
        for sched in schedulers:
            ms = scheds[sched]["avg"]["makespan_ms"]
            delta = 100.0 * (base_ms - ms) / base_ms
            total_delta[sched] += delta
            if delta > 0.5:
                wins[sched] += 1
            elif delta < -0.5:
                losses[sched] += 1
            else:
                ties[sched] += 1
            mark = '+' if delta > 0.5 else ('-' if delta < -0.5 else '=')
            print(f"{delta:>+7.2f}{mark}         ", end="")
        print()

    print()
    print(f"{'— aggregate —':<40s} ", end="")
    for sched in schedulers:
        print(f"{total_delta[sched]/len(data):>+7.2f}          ", end="")
    print()
    print(f"{'— win/tie/loss —':<40s} ", end="")
    for sched in schedulers:
        w = wins[sched]
        t = ties[sched]
        l = losses[sched]
        print(f"{w:>4d}/{t:>2d}/{l:<3d}          ", end="")
    print()


if __name__ == "__main__":
    main()
