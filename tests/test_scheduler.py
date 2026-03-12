"""Tests for scheduling policies."""

import pytest

from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler, make_scheduler
from hops.core.types import Phase

from .conftest import make_test_pipeline


def test_gpipe_all_forwards_before_backwards():
    """GPipe: all forward passes should complete before any backward starts."""
    engine, pipeline, collector = make_test_pipeline(GPipeScheduler())
    pipeline.start_batch(4)
    engine.run()

    forwards = [(r.start_time, r.end_time) for r in collector.computes
                if r.phase == Phase.FORWARD]
    backwards = [(r.start_time, r.end_time) for r in collector.computes
                 if r.phase == Phase.BACKWARD]

    last_forward_end = max(end for _, end in forwards)
    first_backward_start = min(start for start, _ in backwards)
    assert first_backward_start >= last_forward_end - 0.01


def test_1f1b_interleaves():
    """1F1B: the last stage should start backward before all its forwards complete."""
    engine, pipeline, collector = make_test_pipeline(
        OneFOneBScheduler(), num_stages=4)
    pipeline.start_batch(8)
    engine.run()

    last_stage = 3
    fwds = [r for r in collector.computes
            if r.stage_id == last_stage and r.phase == Phase.FORWARD]
    bwds = [r for r in collector.computes
            if r.stage_id == last_stage and r.phase == Phase.BACKWARD]

    assert len(bwds) > 0
    first_bwd_start = min(r.start_time for r in bwds)
    last_fwd_end = max(r.end_time for r in fwds)
    assert first_bwd_start < last_fwd_end


def test_make_scheduler_factory():
    assert isinstance(make_scheduler({"policy": "gpipe"}), GPipeScheduler)
    assert isinstance(make_scheduler({"policy": "1f1b"}), OneFOneBScheduler)


def test_make_scheduler_unknown_raises():
    with pytest.raises(ValueError):
        make_scheduler({"policy": "unknown"})


def test_all_microbatches_complete():
    """Both schedulers should complete all micro-batches."""
    for sched in [GPipeScheduler(), OneFOneBScheduler()]:
        engine, pipeline, collector = make_test_pipeline(sched)
        pipeline.start_batch(4)
        engine.run()
        mb_ids = set(r.microbatch_id for r in collector.computes)
        assert len(mb_ids) == 4
