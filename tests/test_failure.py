"""Tests for failure injection."""

import numpy as np

from hops.core.event_engine import EventEngine
from hops.failure.engine import FailureEngine
from hops.hardware.device import Device
from hops.hardware.topology import Topology
from hops.metrics.collector import MetricsCollector


def _make_failure_setup(device_fail_prob=0.0, check_interval=1.0):
    devices = [Device("gpu0", "gpu", 100, 8192, 1000)]
    topology = Topology(devices, [])
    collector = MetricsCollector()
    engine = EventEngine()
    config = {
        "check_interval": check_interval,
        "device_fail_prob": device_fail_prob,
        "link_fail_prob": 0.0,
        "recovery_time": 2.0,
    }
    failure_engine = FailureEngine(engine, topology, collector, config)
    return engine, failure_engine, collector


def test_no_failures_with_zero_prob():
    np.random.seed(0)
    engine, fe, collector = _make_failure_setup(device_fail_prob=0.0)
    engine.run(until=10.0)
    assert len(collector.failures) == 0


def test_failures_with_high_prob():
    np.random.seed(0)
    engine, fe, collector = _make_failure_setup(device_fail_prob=1.0)
    engine.run(until=5.0)
    assert len(collector.failures) > 0


def test_recovery_clears_failed_state():
    """After failure + recovery, device should be not-failed (before next check)."""
    np.random.seed(0)
    engine, fe, collector = _make_failure_setup(
        device_fail_prob=1.0, check_interval=100.0)

    engine.run(until=100.0)  # first check at t=100, failure injected
    assert fe.is_failed("gpu0")
    engine.run(until=103.0)  # recovery at t=102
    assert not fe.is_failed("gpu0")
