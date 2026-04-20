"""Benchmark pipeline schedules on synthetic L4/A10G configs (no fixture reads)."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy

from hops.config import parse_config
from hops.runtime import build_runtime


def _device(id_: str, gpu: str, node: str, socket: int) -> dict:
    return {"id": id_, "gpu": gpu, "node": node, "socket": socket}


def _stage(device: str, tflop: float, memory_mb: float, weights_mb: float) -> dict:
    return {
        "device": device,
        "weights_mb": weights_mb,
        "compute": {
            "mode": "analytical",
            "tflop": tflop,
            "memory_mb": memory_mb,
            "efficiency": {"compute": 0.72, "memory": 0.82},
            "jitter": {"type": "constant", "value": 0.0},
        },
    }


def _base_template() -> dict:
    return {
        "simulation": {"batches": 1, "microbatches": 8, "seed": 0},
        "pipeline": {
            "schedule": "zero_bubble",
            "precision": "bf16",
            "activation_mb": 40.0,
            "backward_factor": 2.0,
            "backward_split": {"enabled": True, "activation_grad_fraction": 0.5},
            "stages": [],
        },
        "hardware": {
            "devices": [],
            "interconnect": {"same_node": "pcie", "cross_node": "ethernet"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    }


def scenario_pp2_l4_then_a10g() -> dict:
    c = _base_template()
    c["hardware"]["devices"] = [
        _device("d0", "l4", "n0", 0),
        _device("d1", "a10g", "n1", 0),
    ]
    c["pipeline"]["stages"] = [
        _stage("d0", 4.0, 200.0, 4000.0),
        _stage("d1", 4.0, 200.0, 4000.0),
    ]
    return c


def scenario_pp2_a10g_then_l4() -> dict:
    c = scenario_pp2_l4_then_a10g()
    c["hardware"]["devices"] = [
        _device("d0", "a10g", "n0", 0),
        _device("d1", "l4", "n1", 0),
    ]
    c["pipeline"]["stages"] = [
        _stage("d0", 4.0, 200.0, 4000.0),
        _stage("d1", 4.0, 200.0, 4000.0),
    ]
    return c


def scenario_pp2_slow_upstream() -> dict:
    c = scenario_pp2_l4_then_a10g()
    c["pipeline"]["stages"] = [
        _stage("d0", 6.0, 280.0, 4000.0),
        _stage("d1", 3.0, 150.0, 4000.0),
    ]
    return c


def scenario_pp2_fast_then_slow() -> dict:
    c = scenario_pp2_l4_then_a10g()
    c["hardware"]["devices"] = [
        _device("d0", "a10g", "n0", 0),
        _device("d1", "l4", "n1", 0),
    ]
    c["pipeline"]["stages"] = [
        _stage("d0", 3.0, 150.0, 4000.0),
        _stage("d1", 6.0, 280.0, 4000.0),
    ]
    return c


def scenario_pp4_mixed_order(order: str) -> dict:
    c = _base_template()
    if order == "l4_l4_a10g_a10g":
        devs = [
            _device("d0", "l4", "n0", 0),
            _device("d1", "l4", "n0", 1),
            _device("d2", "a10g", "n0", 2),
            _device("d3", "a10g", "n0", 3),
        ]
        stages = [
            _stage("d0", 2.5, 120.0, 2500.0),
            _stage("d1", 2.5, 120.0, 2500.0),
            _stage("d2", 2.5, 120.0, 2500.0),
            _stage("d3", 2.5, 120.0, 2500.0),
        ]
    elif order == "a10g_l4_a10g_l4":
        devs = [
            _device("d0", "a10g", "n0", 0),
            _device("d1", "l4", "n0", 1),
            _device("d2", "a10g", "n0", 2),
            _device("d3", "l4", "n0", 3),
        ]
        stages = [
            _stage("d0", 2.5, 120.0, 2500.0),
            _stage("d1", 2.5, 120.0, 2500.0),
            _stage("d2", 2.5, 120.0, 2500.0),
            _stage("d3", 2.5, 120.0, 2500.0),
        ]
    else:
        raise ValueError(order)
    c["hardware"]["devices"] = devs
    c["pipeline"]["stages"] = stages
    return c


def scenario_pp4_skewed_flops() -> dict:
    c = scenario_pp4_mixed_order("l4_l4_a10g_a10g")
    c["pipeline"]["stages"] = [
        _stage("d0", 4.0, 200.0, 2500.0),
        _stage("d1", 3.0, 160.0, 2500.0),
        _stage("d2", 2.0, 120.0, 2500.0),
        _stage("d3", 1.5, 100.0, 2500.0),
    ]
    return c


def scenario_pp3_hot_middle() -> dict:
    c = _base_template()
    c["hardware"]["devices"] = [
        _device("d0", "l4", "n0", 0),
        _device("d1", "l4", "n0", 1),
        _device("d2", "a10g", "n1", 0),
    ]
    c["pipeline"]["stages"] = [
        _stage("d0", 1.2, 80.0, 3000.0),
        _stage("d1", 8.0, 400.0, 3000.0),
        _stage("d2", 2.0, 120.0, 3000.0),
    ]
    return c


SCENARIOS: dict[str, callable[[], dict]] = {
    "pp2_l4_a10g_uniform": scenario_pp2_l4_then_a10g,
    "pp2_a10g_l4_uniform": scenario_pp2_a10g_then_l4,
    "pp2_l4_a10g_skew": scenario_pp2_slow_upstream,
    "pp2_a10g_l4_skew_fast_then_slow": scenario_pp2_fast_then_slow,
    "pp4_l4l4a10ga10g_uniform": lambda: scenario_pp4_mixed_order("l4_l4_a10g_a10g"),
    "pp4_a10gl4a10gl4_uniform": lambda: scenario_pp4_mixed_order("a10g_l4_a10g_l4"),
    "pp4_l4l4a10ga10g_skew_flops": scenario_pp4_skewed_flops,
    "pp3_l4_l4_a10g_hot_middle": scenario_pp3_hot_middle,
}


def run_case(raw: dict, schedule: str, microbatches: int) -> dict:
    cfg = deepcopy(raw)
    cfg["simulation"]["microbatches"] = microbatches
    cfg["pipeline"]["schedule"] = schedule
    app = parse_config(cfg)
    rt = build_runtime(app)
    for _ in range(rt.num_batches):
        rt.pipeline.start_batch(rt.num_microbatches)
        rt.engine.run(stop_condition=lambda: rt.pipeline.batch_complete)
    s = rt.reporter.summary_model()
    util = s.utilization.per_stage or {}
    util_str = ",".join(f"s{k}:{v:.3f}" for k, v in sorted(util.items()))
    peak_mem = max(s.memory.peak_per_device_mb.values()) if s.memory.peak_per_device_mb else 0.0
    return {
        "throughput_per_s": round(s.throughput.per_s, 4),
        "bubble_ratio_pct": round(100.0 * s.bubble_ratio, 3),
        "latency_mean_ms": round(s.latency_ms.mean_ms or 0.0, 4),
        "peak_mem_mb": round(peak_mem, 1),
        "util_per_stage": util_str,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", action="append", help="repeatable; default = all")
    p.add_argument("--schedules", default="gpipe,1f1b,zero_bubble,heterogeneous_hops")
    p.add_argument("--microbatches", default="8,12,16,24")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    names = args.scenario or sorted(SCENARIOS.keys())
    schedules = [x.strip() for x in args.schedules.split(",") if x.strip()]
    mbs = [int(x.strip()) for x in args.microbatches.split(",") if x.strip()]
    rows = []
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario {name!r}", file=sys.stderr)
            sys.exit(1)
        base = SCENARIOS[name]()
        for mb in mbs:
            for sch in schedules:
                row = {
                    "scenario": name,
                    "microbatches": mb,
                    "schedule": sch,
                    **run_case(base, sch, mb),
                }
                rows.append(row)
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
    else:
        hdr = (
            "scenario\tmb\tschedule\tthr_per_s\tbubble%\tlatency_ms\tpeak_mem_mb\tutil"
        )
        print(hdr)
        for r in rows:
            print(
                f"{r['scenario']}\t{r['microbatches']}\t{r['schedule']}\t"
                f"{r['throughput_per_s']}\t{r['bubble_ratio_pct']}\t"
                f"{r['latency_mean_ms']}\t{r['peak_mem_mb']}\t{r['util_per_stage']}"
            )


if __name__ == "__main__":
    main()
