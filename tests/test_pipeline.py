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


def test_partial_run_does_not_count_completed_microbatches():
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=2, compute_time=10.0, seed=0)
    pipeline.start_batch(2)
    engine.run(until=10.0)

    assert collector.completed_microbatches == 0
    assert collector.e2e_latencies() == []


def test_microbatch_completion_records_true_end_to_end_latency():
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=2, compute_time=10.0, seed=0)
    pipeline.start_batch(1)
    engine.run()

    assert collector.completed_microbatches == 1
    assert collector.e2e_latencies() == [60.0]


def test_gpipe_bubble_ratio_matches_closed_form():
    num_stages = 4
    num_microbatches = 8
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=num_stages, compute_time=1.0, seed=0)
    pipeline.start_batch(num_microbatches)
    engine.run()

    expected = (num_stages - 1) / (num_stages - 1 + num_microbatches)
    assert abs(collector.bubble_ratio() - expected) < 1e-9


def test_1f1b_peak_in_flight_per_stage_is_bounded_by_stage_count():
    num_stages = 4
    engine, pipeline, collector = make_test_pipeline(
        OneFOneBScheduler(), num_stages=num_stages, compute_time=1.0, seed=0)
    pipeline.start_batch(8)
    engine.run()

    peaks = collector.peak_in_flight_per_stage()
    assert peaks
    assert all(count <= num_stages for count in peaks.values())


def test_gpipe_stage_occupancy_and_idle_intervals_match_trace():
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=2, compute_time=1.0, seed=0)
    pipeline.start_batch(2)
    engine.run()

    assert collector.stage_occupancy_intervals() == {
        0: [(0.0, 2.0), (5.0, 9.0)],
        1: [(1.0, 7.0)],
    }
    assert collector.stage_idle_intervals() == {
        0: [(2.0, 5.0)],
        1: [(0.0, 1.0), (7.0, 9.0)],
    }


def test_1f1b_last_stage_sequence_matches_warmup_steady_flush_pattern():
    engine, pipeline, collector = make_test_pipeline(
        OneFOneBScheduler(), num_stages=2, compute_time=1.0, seed=0)
    pipeline.start_batch(2)
    engine.run()

    last_stage_records = [
        (record.phase, record.start_time, record.end_time)
        for record in collector.computes
        if record.stage_id == 1
    ]
    assert last_stage_records == [
        (Phase.FORWARD, 1.0, 2.0),
        (Phase.BACKWARD, 2.0, 4.0),
        (Phase.FORWARD, 4.0, 5.0),
        (Phase.BACKWARD, 5.0, 7.0),
    ]
