"""Tests for scheduling policies."""

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler, make_scheduler
from hops.core.types import Phase
from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant
from hops.metrics.collector import MetricsCollector


def _run_pipeline(scheduler, num_stages=4, num_microbatches=4, compute_time=5.0):
    """Run a pipeline and return the collector."""
    np.random.seed(42)
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
    pipeline.start_batch(num_microbatches)
    engine.run()
    return collector


def test_gpipe_all_forwards_before_backwards():
    """GPipe: all forward passes should complete before any backward starts."""
    collector = _run_pipeline(GPipeScheduler())

    forwards = [(r.start_time, r.end_time) for r in collector.computes
                if r.phase == Phase.FORWARD]
    backwards = [(r.start_time, r.end_time) for r in collector.computes
                 if r.phase == Phase.BACKWARD]

    last_forward_end = max(end for _, end in forwards)
    first_backward_start = min(start for start, _ in backwards)

    # In GPipe, backward starts at the last stage right after its forward ends.
    # So first_backward_start == last_forward_end for the last stage.
    assert first_backward_start >= last_forward_end - 0.01


def test_1f1b_interleaves():
    """1F1B: the last stage should start backward before all its forwards complete."""
    collector = _run_pipeline(OneFOneBScheduler(), num_stages=4, num_microbatches=8)

    # Last stage (3) has warmup=1, so it should start backward after 1 forward,
    # well before all 8 forwards complete there.
    last_stage = 3
    fwds = [r for r in collector.computes
            if r.stage_id == last_stage and r.phase == Phase.FORWARD]
    bwds = [r for r in collector.computes
            if r.stage_id == last_stage and r.phase == Phase.BACKWARD]

    assert len(bwds) > 0
    first_bwd_start = min(r.start_time for r in bwds)
    last_fwd_end = max(r.end_time for r in fwds)
    # Backward should interleave with forwards at this stage
    assert first_bwd_start < last_fwd_end


def test_make_scheduler_factory():
    s = make_scheduler({"policy": "gpipe"})
    assert isinstance(s, GPipeScheduler)
    s = make_scheduler({"policy": "1f1b"})
    assert isinstance(s, OneFOneBScheduler)


def test_make_scheduler_unknown_raises():
    try:
        make_scheduler({"policy": "unknown"})
        assert False
    except ValueError:
        pass


def test_all_microbatches_complete():
    """Both schedulers should complete all micro-batches."""
    for sched in [GPipeScheduler(), OneFOneBScheduler()]:
        collector = _run_pipeline(sched, num_microbatches=4)
        mb_ids = set(r.microbatch_id for r in collector.computes)
        assert len(mb_ids) == 4
