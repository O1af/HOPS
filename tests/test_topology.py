"""Tests for hardware topology."""

from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Topology
from hops.latency.distributions import Constant


def test_device_from_yaml():
    d = Device.from_yaml({
        "id": "gpu0", "kind": "gpu", "flops": 100.0,
        "memory_mb": 8192, "memory_bandwidth_gbps": 1000, "numa_node": 0,
    })
    assert d.id == "gpu0"
    assert d.flops == 100.0


def test_link_transfer_time_positive():
    link = Link("a", "b", bandwidth_gbps=100.0, base_latency_us=5.0,
                jitter=Constant(0.0))
    t = link.sample_transfer_time(50.0)
    assert t > 0.0


def test_link_transfer_time_scales_with_size():
    link = Link("a", "b", bandwidth_gbps=100.0, base_latency_us=0.0,
                jitter=Constant(0.0))
    t1 = link.sample_transfer_time(10.0)
    t2 = link.sample_transfer_time(20.0)
    assert abs(t2 - 2 * t1) < 1e-10


def test_topology_from_yaml():
    config = {
        "devices": [
            {"id": "gpu0", "kind": "gpu", "flops": 100, "memory_mb": 8192,
             "memory_bandwidth_gbps": 1000},
            {"id": "gpu1", "kind": "gpu", "flops": 100, "memory_mb": 8192,
             "memory_bandwidth_gbps": 1000},
        ],
        "links": [
            {"src": "gpu0", "dst": "gpu1", "bandwidth_gbps": 100,
             "base_latency_us": 1.0, "jitter": {"type": "constant", "value": 0.0}},
        ],
    }
    topo = Topology.from_yaml(config)
    assert topo.device("gpu0").kind == "gpu"
    assert topo.link("gpu0", "gpu1").bandwidth_gbps == 100


def test_same_device_link_is_free():
    topo = Topology(
        [Device("gpu0", "gpu", 100, 8192, 1000)],
        [],
    )
    link = topo.link("gpu0", "gpu0")
    assert link.sample_transfer_time(100.0) == 0.0
