"""End-to-end integration tests."""

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

    np.random.seed(config["simulation"]["seed"])
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
                        config["hardware"].get("activation_size_mb", 50.0))

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
