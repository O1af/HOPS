"""Heterogeneous scheduler evaluation harness.

Usage:
    uv run python experiments/hetero_scheduler/evaluate.py
    uv run python experiments/hetero_scheduler/evaluate.py --schedulers zero_bubble hetero_critical_path
    uv run python experiments/hetero_scheduler/evaluate.py --configs hetero_pp4.yaml hetero_pp4_big.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hops.config import parse_config
from hops.runtime import build_runtime


def _register_custom_schedulers() -> None:
    try:
        importlib.import_module("hops.core.hetero_schedulers")
    except ModuleNotFoundError:
        pass


@dataclass
class RunResult:
    config: str
    scheduler: str
    seed: int
    makespan_ms: float
    throughput_per_s: float
    bubble_ratio: float
    p50_latency_ms: float | None
    p99_latency_ms: float | None
    per_stage_util: dict[int, float]
    per_device_util: dict[str, float]
    compute_ms: float
    transfer_ms: float
    peak_memory_per_device: dict[str, float]


def run_once(config_path: Path, scheduler: str, seed: int) -> RunResult:
    with open(config_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = dict(raw)
    raw["pipeline"] = {**raw["pipeline"], "schedule": scheduler}
    raw["simulation"] = {**raw["simulation"], "seed": seed}
    raw["output"] = {}

    config = parse_config(raw)
    runtime = build_runtime(config)
    for _ in range(runtime.num_batches):
        runtime.pipeline.start_batch(runtime.num_microbatches)
        runtime.engine.run(stop_condition=lambda: runtime.pipeline.batch_complete)

    summary = runtime.reporter.summary_model()

    return RunResult(
        config=config_path.name,
        scheduler=scheduler,
        seed=seed,
        makespan_ms=summary.time_ms.makespan_ms,
        throughput_per_s=summary.throughput.per_s,
        bubble_ratio=summary.bubble_ratio,
        p50_latency_ms=summary.latency_ms.p50_ms,
        p99_latency_ms=summary.latency_ms.p99_ms,
        per_stage_util=dict(summary.utilization.per_stage),
        per_device_util=dict(summary.utilization.per_device),
        compute_ms=summary.time_ms.compute_ms,
        transfer_ms=summary.time_ms.transfer_ms,
        peak_memory_per_device=dict(summary.memory.peak_per_device_mb),
    )


def average_runs(runs: list[RunResult]) -> dict:
    return {
        "makespan_ms": statistics.mean(r.makespan_ms for r in runs),
        "throughput_per_s": statistics.mean(r.throughput_per_s for r in runs),
        "bubble_ratio": statistics.mean(r.bubble_ratio for r in runs),
        "p50_latency_ms": statistics.mean(r.p50_latency_ms for r in runs if r.p50_latency_ms is not None),
        "p99_latency_ms": statistics.mean(r.p99_latency_ms for r in runs if r.p99_latency_ms is not None),
        "per_device_util_mean": {
            k: statistics.mean(r.per_device_util.get(k, 0.0) for r in runs)
            for k in runs[0].per_device_util
        },
        "makespan_std": statistics.pstdev(r.makespan_ms for r in runs) if len(runs) > 1 else 0.0,
    }


def evaluate(configs: list[Path], schedulers: list[str], seeds: list[int]) -> dict:
    _register_custom_schedulers()
    results: dict[str, dict[str, dict]] = {}
    for config_path in configs:
        cfg_name = config_path.name
        results[cfg_name] = {}
        for sched in schedulers:
            runs = [run_once(config_path, sched, seed) for seed in seeds]
            avg = average_runs(runs)
            results[cfg_name][sched] = {
                "avg": avg,
                "runs": [asdict(r) for r in runs],
            }
    return results


def print_report(results: dict, baseline: str) -> None:
    for cfg_name, scheds in results.items():
        print(f"\n===== {cfg_name} =====")
        base_tp = scheds[baseline]["avg"]["throughput_per_s"]
        base_ms = scheds[baseline]["avg"]["makespan_ms"]
        header = (
            f"{'scheduler':<28s} {'makespan_ms':>12s} {'Δ_vs_base_%':>12s} "
            f"{'throughput/s':>13s} {'bubble':>8s} {'util_mean':>10s}"
        )
        print(header)
        print("-" * len(header))
        for sched, entry in scheds.items():
            avg = entry["avg"]
            delta = 100.0 * (base_ms - avg["makespan_ms"]) / base_ms if base_ms else 0.0
            util_mean = statistics.mean(avg["per_device_util_mean"].values())
            marker = " *" if sched == baseline else ""
            print(
                f"{sched:<28s} {avg['makespan_ms']:>12.2f} {delta:>11.2f}% "
                f"{avg['throughput_per_s']:>13.2f} {avg['bubble_ratio']:>7.2%} "
                f"{util_mean:>9.2%}{marker}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["hetero_pp4.yaml"],
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=None,
        help="Scheduler names; default runs all registered ones.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
    )
    parser.add_argument(
        "--baseline",
        default="zero_bubble",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    _register_custom_schedulers()

    here = Path(__file__).parent
    config_paths = [(here / c) if not Path(c).is_absolute() else Path(c) for c in args.configs]

    if args.schedulers is None:
        from hops.core.scheduler import _SCHEDULER_REGISTRY
        schedulers = ["gpipe", "1f1b", "zero_bubble"] + [
            s for s in _SCHEDULER_REGISTRY
            if s not in {"gpipe", "1f1b", "zero_bubble"}
        ]
    else:
        schedulers = args.schedulers

    results = evaluate(config_paths, schedulers, args.seeds)
    print_report(results, args.baseline)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
