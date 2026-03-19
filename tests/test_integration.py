"""End-to-end integration tests."""

import builtins
import csv
import importlib
import json
import sys

import numpy as np
import yaml

from hops.config import parse_config
from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler
from hops.core.types import Phase
from hops.runtime import build_runtime

from .conftest import make_canonical_config, make_test_pipeline


def test_default_config_runs():
    """Load default.yaml and run a full simulation."""
    with open("configs/default.yaml") as f:
        raw = yaml.safe_load(f)

    runtime = build_runtime(parse_config(raw))
    for _ in range(runtime.num_batches):
        runtime.pipeline.start_batch(runtime.num_microbatches)
        runtime.engine.run()

    assert runtime.collector.completed_microbatches == runtime.num_batches * runtime.num_microbatches
    assert runtime.collector.throughput() > 0
    assert 0 < runtime.collector.bubble_ratio() < 1


def test_1f1b_less_bubbles_than_gpipe():
    """1F1B should have lower bubble ratio than GPipe on the same config."""
    def run_with_scheduler(scheduler):
        engine, pipeline, collector = make_test_pipeline(scheduler)
        pipeline.start_batch(8)
        engine.run()
        return collector.bubble_ratio()

    gpipe_bubble = run_with_scheduler(GPipeScheduler())
    onefb_bubble = run_with_scheduler(OneFOneBScheduler())

    assert onefb_bubble <= gpipe_bubble, (
        f"1F1B bubble {onefb_bubble:.2%} should be <= GPipe {gpipe_bubble:.2%}")


def test_main_no_viz_does_not_require_matplotlib(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(make_canonical_config()), encoding="utf-8")

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("matplotlib"):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--config", str(config_path), "--no-viz"],
    )
    for module_name in ("main", "hops.viz.dashboard", "hops.viz.timeline"):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("main")
    module.main()


def test_topology_fabric_and_derived_latency_run_together():
    config = {
        "simulation": {"batches": 1, "microbatches": 2, "seed": 0},
        "pipeline": {
            "schedule": "gpipe",
            "precision": "fp32",
            "activation_mb": 10.0,
            "backward_factor": 2.0,
            "stages": [
                {
                    "device": "n0_gpu0",
                    "weights_mb": 0.0,
                    "compute": {
                        "mode": "analytical",
                        "tflop": 8.0,
                        "memory_mb": 200.0,
                    },
                },
                {
                    "device": "n1_gpu0",
                    "weights_mb": 0.0,
                    "compute": {
                        "mode": "analytical",
                        "tflop": 12.0,
                        "memory_mb": 200.0,
                    },
                },
            ],
        },
        "hardware": {
            "devices": [
                {
                    "id": "n0_gpu0",
                    "gpu": "h100",
                    "node": "n0",
                    "socket": 0,
                },
                {
                    "id": "n1_gpu0",
                    "gpu": "a100",
                    "node": "n1",
                    "socket": 0,
                },
            ],
            "interconnect": {
                "same_node": "nvlink",
                "cross_node": "infiniband",
            },
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    }

    runtime = build_runtime(parse_config(config))
    runtime.pipeline.start_batch(runtime.num_microbatches)
    runtime.engine.run()

    collector = runtime.collector
    assert collector.completed_microbatches == 2
    stage0_forward = next(
        r for r in collector.computes
        if r.stage_id == 0 and r.phase == Phase.FORWARD
    )
    stage1_forward = next(
        r for r in collector.computes
        if r.stage_id == 1 and r.phase == Phase.FORWARD
    )
    assert (stage1_forward.end_time - stage1_forward.start_time) > (
        stage0_forward.end_time - stage0_forward.start_time
    )


def test_main_writes_summary_json_and_trace_csv(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.json"
    trace_path = tmp_path / "trace.csv"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(make_canonical_config()), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--config",
            str(config_path),
            "--no-viz",
            "--summary-json",
            str(summary_path),
            "--trace-csv",
            str(trace_path),
        ],
    )
    sys.modules.pop("main", None)
    module = importlib.import_module("main")
    module.main()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["completed_microbatches"] == 1
    assert "utilization" in summary
    assert "per_device" in summary["utilization"]

    with trace_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert any(row["event_type"] == "compute" for row in rows)
