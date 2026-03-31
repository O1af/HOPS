"""Tests for failure injection."""

import numpy as np

from hops.config import FailureConfig
from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import GPipeScheduler
from hops.core.types import Event, EventKind
from hops.failure.engine import FailureEngine
from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant
from hops.metrics.collector import MetricsCollector


def _make_failure_setup(device_fail_prob=0.0, check_interval=1.0, seed=0):
    rng = np.random.default_rng(seed)
    devices = [Device("gpu0", "gpu", 8192)]
    topology = Topology(devices, [])
    collector = MetricsCollector()
    engine = EventEngine()
    config = {
        "check_interval_ms": check_interval,
        "device_failure_probability": device_fail_prob,
        "link_failure_probability": 0.0,
        "recovery_time_ms": 2.0,
    }
    failure_engine = FailureEngine(engine, topology, collector, FailureConfig(enabled=True, **config), rng=rng)
    return engine, failure_engine, collector


def test_no_failures_with_zero_prob():
    engine, fe, collector = _make_failure_setup(device_fail_prob=0.0)
    engine.run(until=10.0)
    assert len(collector.failures) == 0


def test_failures_with_high_prob():
    engine, fe, collector = _make_failure_setup(device_fail_prob=1.0)
    engine.run(until=5.0)
    assert len(collector.failures) > 0


def test_recovery_clears_failed_state():
    """After failure + recovery, device should be not-failed (before next check)."""
    engine, fe, collector = _make_failure_setup(
        device_fail_prob=1.0, check_interval=100.0)

    engine.run(until=100.0)  # first check at t=100, failure injected
    assert fe.is_failed("gpu0")
    engine.run(until=103.0)  # recovery at t=102
    assert not fe.is_failed("gpu0")


def test_device_failure_delays_compute_start():
    rng = np.random.default_rng(0)
    devices = [Device("gpu0", "gpu", 8192)]
    topology = Topology(devices, [])
    collector = MetricsCollector()
    engine = EventEngine()
    pipeline = Pipeline(
        [Stage(0, "gpu0")],
        engine,
        topology,
        ComputeModel({0: Constant(10.0)}),
        GPipeScheduler(),
        collector,
        activation_size_mb=0.0,
        rng=rng,
    )
    failure_engine = FailureEngine(
        engine,
        topology,
        collector,
        FailureConfig(
            enabled=True,
            check_interval_ms=1000.0,
            device_failure_probability=0.0,
            link_failure_probability=0.0,
            recovery_time_ms=10.0,
        ),
        rng=rng,
    )
    pipeline.set_failure_engine(failure_engine)

    engine.schedule(Event(
        time=0.0,
        kind=EventKind.FAILURE,
        payload={"target_type": "device", "device_id": "gpu0"},
    ))
    pipeline.start_batch(1)
    engine.run(stop_condition=lambda: pipeline.batch_complete)

    assert collector.completed_microbatches == 1
    assert collector.computes[0].start_time >= 10.0


def test_link_failure_delays_transfer_start():
    rng = np.random.default_rng(0)
    devices = [Device("gpu0", "gpu", 8192), Device("gpu1", "gpu", 8192)]
    topology = Topology(devices, [
        Link("gpu0", "gpu1", 1000, 0.0, Constant(0.0)),
        Link("gpu1", "gpu0", 1000, 0.0, Constant(0.0)),
    ])
    collector = MetricsCollector()
    engine = EventEngine()
    pipeline = Pipeline(
        [Stage(0, "gpu0"), Stage(1, "gpu1")],
        engine,
        topology,
        ComputeModel({0: Constant(10.0), 1: Constant(10.0)}),
        GPipeScheduler(),
        collector,
        activation_size_mb=10.0,
        rng=rng,
    )
    failure_engine = FailureEngine(
        engine,
        topology,
        collector,
        FailureConfig(
            enabled=True,
            check_interval_ms=1000.0,
            device_failure_probability=0.0,
            link_failure_probability=0.0,
            recovery_time_ms=15.0,
        ),
        rng=rng,
    )
    pipeline.set_failure_engine(failure_engine)

    engine.schedule(Event(
        time=0.0,
        kind=EventKind.FAILURE,
        payload={"target_type": "link", "src": "gpu0", "dst": "gpu1"},
    ))
    pipeline.start_batch(1)
    engine.run(until=40.0)

    assert collector.transfers
    assert collector.transfers[0].start_time >= 15.0


def test_failure_enabled_batch_run_stops_when_pipeline_finishes():
    rng = np.random.default_rng(0)
    devices = [Device("gpu0", "gpu", 8192), Device("gpu1", "gpu", 8192)]
    links = [
        Link("gpu0", "gpu1", 1000, 0.0, Constant(0.0)),
        Link("gpu1", "gpu0", 1000, 0.0, Constant(0.0)),
    ]
    topology = Topology(devices, links)
    collector = MetricsCollector()
    engine = EventEngine()
    pipeline = Pipeline(
        [Stage(0, "gpu0"), Stage(1, "gpu1")],
        engine,
        topology,
        ComputeModel({0: Constant(5.0), 1: Constant(5.0)}),
        GPipeScheduler(),
        collector,
        activation_size_mb=0.0,
        rng=rng,
    )
    failure_engine = FailureEngine(
        engine,
        topology,
        collector,
        FailureConfig(
            enabled=True,
            check_interval_ms=1.0,
            device_failure_probability=1.0,
            link_failure_probability=1.0,
            recovery_time_ms=100.0,
        ),
        rng=rng,
    )
    pipeline.set_failure_engine(failure_engine)

    pipeline.start_batch(1)
    engine.run(stop_condition=lambda: pipeline.batch_complete)

    assert pipeline.batch_complete
    assert collector.completed_microbatches == 1
    assert engine.pending() > 0
    assert len(collector.failures) > 0
