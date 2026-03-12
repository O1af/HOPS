"""Tests for the pipeline model."""

import numpy as np

from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler
from hops.core.types import Phase

from .conftest import make_test_pipeline


def test_two_stage_completes_all_microbatches():
    """2-stage pipeline with 2 MBs should complete all forward and backward passes."""
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=2, seed=0)
    pipeline.start_batch(2)
    engine.run()

    assert len(collector.computes) == 8  # 2 MBs * 2 stages * 2 phases

    fwd_count = sum(1 for r in collector.computes if r.phase == Phase.FORWARD)
    bwd_count = sum(1 for r in collector.computes if r.phase == Phase.BACKWARD)
    assert fwd_count == 4
    assert bwd_count == 4


def test_deterministic_timing():
    """With constant latency and zero transfer time, timing should be predictable."""
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=2, compute_time=10.0, seed=0)
    pipeline.start_batch(1)
    engine.run()

    # 1 MB, 2 stages: fwd_s0(10) + fwd_s1(10) + bwd_s1(20) + bwd_s0(20)
    latencies = collector.e2e_latencies()
    assert len(latencies) == 1
    assert abs(latencies[0] - 60.0) < 0.01


def test_all_microbatches_in_collector():
    engine, pipeline, collector = make_test_pipeline(
        OneFOneBScheduler(), num_stages=2, seed=0)
    pipeline.start_batch(4)
    engine.run()

    mb_ids = set(r.microbatch_id for r in collector.computes)
    assert len(mb_ids) == 4


def test_utilization_bounded():
    """Stage utilization should be between 0 and 1."""
    engine, pipeline, collector = make_test_pipeline(
        OneFOneBScheduler(), num_stages=4, seed=0)
    pipeline.start_batch(8)
    engine.run()

    util = collector.per_stage_utilization()
    for stage_id, u in util.items():
        assert 0.0 < u <= 1.0, f"Stage {stage_id} utilization {u} out of bounds"
