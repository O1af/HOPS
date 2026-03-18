"""End-to-end integration tests."""

import builtins
import importlib
import sys

import numpy as np
import yaml

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.metrics.collector import MetricsCollector

from .conftest import make_test_pipeline


def test_default_config_runs():
    """Load default.yaml and run a full simulation."""
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(config["simulation"]["seed"])
    topology = Topology.from_yaml(config["hardware"])
    compute_model = ComputeModel.from_yaml(config["pipeline"])
    collector = MetricsCollector()
    engine = EventEngine()

    stages = [
        Stage(id=s["id"], device_id=s["device"])
        for s in config["pipeline"]["stages"]
    ]
    pipeline = Pipeline(stages, engine, topology, compute_model,
                        OneFOneBScheduler(), collector,
                        config["hardware"].get("activation_size_mb", 50.0),
                        rng=rng)

    num_batches = config["simulation"]["num_batches"]
    num_mb = config["simulation"]["num_microbatches"]
    for _ in range(num_batches):
        pipeline.start_batch(num_mb)
        engine.run()

    assert collector.completed_microbatches == num_batches * num_mb
    assert collector.throughput() > 0
    assert 0 < collector.bubble_ratio() < 1


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
    config_path.write_text(
        yaml.safe_dump({
            "simulation": {"num_microbatches": 1, "num_batches": 1, "seed": 0},
            "pipeline": {
                "stages": [{
                    "id": 0,
                    "device": "gpu0",
                    "compute_latency": {"type": "constant", "value": 1.0},
                }],
            },
            "scheduler": {"policy": "gpipe"},
            "hardware": {
                "devices": [{"id": "gpu0", "kind": "gpu", "memory_mb": 8192}],
                "activation_size_mb": 0.0,
            },
        }),
        encoding="utf-8",
    )

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
