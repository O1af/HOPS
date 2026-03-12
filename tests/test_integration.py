"""End-to-end integration tests."""

import numpy as np
import yaml

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.metrics.collector import MetricsCollector


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
        Stage(id=s["id"], device_id=s["device"],
              num_layers=s.get("num_layers", 1))
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

    assert len(collector._mb_end_times) == num_batches * num_mb
    assert collector.throughput() > 0
    assert 0 < collector.bubble_ratio() < 1


def test_1f1b_less_bubbles_than_gpipe():
    """1F1B should have lower bubble ratio than GPipe on the same config."""
    from hops.hardware.device import Device
    from hops.hardware.network import Link
    from hops.latency.distributions import Constant

    def run_with_scheduler(scheduler):
        np.random.seed(42)
        n = 4
        devices = [Device(f"gpu{i}", "gpu", 100, 8192, 1000) for i in range(n)]
        links = []
        for i in range(n - 1):
            links.append(Link(f"gpu{i}", f"gpu{i+1}", 900, 0.0, Constant(0.0)))
            links.append(Link(f"gpu{i+1}", f"gpu{i}", 900, 0.0, Constant(0.0)))
        topology = Topology(devices, links)
        dists = {i: Constant(5.0) for i in range(n)}
        compute_model = ComputeModel(dists)
        collector = MetricsCollector()
        engine = EventEngine()
        stages = [Stage(i, f"gpu{i}") for i in range(n)]
        pipeline = Pipeline(stages, engine, topology, compute_model,
                            scheduler, collector, 0.0)
        pipeline.start_batch(8)
        engine.run()
        return collector.bubble_ratio()

    gpipe_bubble = run_with_scheduler(GPipeScheduler())
    onefb_bubble = run_with_scheduler(OneFOneBScheduler())

    assert onefb_bubble <= gpipe_bubble, (
        f"1F1B bubble {onefb_bubble:.2%} should be <= GPipe {gpipe_bubble:.2%}")
