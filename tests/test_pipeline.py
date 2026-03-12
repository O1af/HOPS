"""Tests for the pipeline model."""

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler
from hops.core.types import Phase
from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant
from hops.metrics.collector import MetricsCollector


def _make_simple_pipeline(scheduler, num_stages=2, compute_time=5.0):
    """Create a minimal pipeline for testing."""
    devices = [Device(f"gpu{i}", "gpu", 100, 8192, 1000) for i in range(num_stages)]
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
                        collector, activation_size_mb=0.0)
    return engine, pipeline, collector


def test_two_stage_completes_all_microbatches():
    """2-stage pipeline with 2 MBs should complete all forward and backward passes."""
    np.random.seed(0)
    engine, pipeline, collector = _make_simple_pipeline(GPipeScheduler())
    pipeline.start_batch(2)
    engine.run()

    # Each MB should have 2 forward + 2 backward computes = 4 per MB
    assert len(collector.computes) == 8  # 2 MBs * 2 stages * 2 phases

    # Check all phases present
    fwd_count = sum(1 for r in collector.computes if r.phase == Phase.FORWARD)
    bwd_count = sum(1 for r in collector.computes if r.phase == Phase.BACKWARD)
    assert fwd_count == 4
    assert bwd_count == 4


def test_deterministic_timing():
    """With constant latency and zero transfer time, timing should be predictable."""
    np.random.seed(0)
    engine, pipeline, collector = _make_simple_pipeline(
        GPipeScheduler(), num_stages=2, compute_time=10.0)
    pipeline.start_batch(1)
    engine.run()

    # 1 MB, 2 stages: fwd_s0(10) + transfer(0) + fwd_s1(10) + bwd_s1(20) + transfer(0) + bwd_s0(20)
    latencies = collector.e2e_latencies()
    assert len(latencies) == 1
    assert abs(latencies[0] - 60.0) < 0.01


def test_all_microbatches_in_collector():
    np.random.seed(0)
    engine, pipeline, collector = _make_simple_pipeline(OneFOneBScheduler())
    pipeline.start_batch(4)
    engine.run()

    mb_ids = set(r.microbatch_id for r in collector.computes)
    assert len(mb_ids) == 4


def test_utilization_bounded():
    """Stage utilization should be between 0 and 1."""
    np.random.seed(0)
    engine, pipeline, collector = _make_simple_pipeline(
        OneFOneBScheduler(), num_stages=4, compute_time=5.0)
    pipeline.start_batch(8)
    engine.run()

    util = collector.per_stage_utilization()
    for stage_id, u in util.items():
        assert 0.0 < u <= 1.0, f"Stage {stage_id} utilization {u} out of bounds"
