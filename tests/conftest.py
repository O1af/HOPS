"""Shared test fixtures for HOPS."""

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import Scheduler
from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant
from hops.metrics.collector import MetricsCollector


def make_canonical_config(*, batches: int = 1, microbatches: int = 1) -> dict:
    return {
        "simulation": {
            "batches": batches,
            "microbatches": microbatches,
            "seed": 0,
        },
        "pipeline": {
            "schedule": "gpipe",
            "precision": "fp32",
            "activation_mb": 0.0,
            "backward_factor": 2.0,
            "stages": [
                {
                    "device": "gpu0",
                    "weights_mb": 0.0,
                    "compute": {
                        "mode": "explicit",
                        "distribution": {"type": "constant", "value": 1.0},
                    },
                }
            ],
        },
        "hardware": {
            "devices": [
                {"id": "gpu0", "gpu": "a100", "node": "node0", "socket": 0}
            ],
            "interconnect": {"same_node": "nvlink", "cross_node": "infiniband"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    }


def make_test_pipeline(scheduler: Scheduler, *, num_stages: int = 4,
                       compute_time: float = 5.0, seed: int = 42):
    """Build a simple homogeneous pipeline for testing.

    Returns (engine, pipeline, collector).
    """
    rng = np.random.default_rng(seed)
    devices = [Device(f"gpu{i}", "gpu", 8192) for i in range(num_stages)]
    links = []
    for i in range(num_stages - 1):
        links.append(Link(f"gpu{i}", f"gpu{i+1}", 900, 0.0, Constant(0.0)))
        links.append(Link(f"gpu{i+1}", f"gpu{i}", 900, 0.0, Constant(0.0)))

    topology = Topology(devices, links)
    dists = {i: Constant(compute_time) for i in range(num_stages)}
    compute_model = ComputeModel(dists)
    collector = MetricsCollector()
    engine = EventEngine()

    stages = [Stage(i, f"gpu{i}") for i in range(num_stages)]
    pipeline = Pipeline(stages, engine, topology, compute_model, scheduler,
                        collector, activation_size_mb=0.0, rng=rng)
    return engine, pipeline, collector
