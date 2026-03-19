"""Tests for hardware topology."""

import numpy as np

from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Locality, Topology
from hops.latency.distributions import Constant


def test_device_from_yaml():
    d = Device.from_yaml({
        "id": "gpu0", "kind": "gpu", "flops": 100.0,
        "memory_mb": 8192, "memory_bandwidth_gbps": 1000, "numa_node": 0,
    })
    assert d.id == "gpu0"
    assert d.memory_mb == 8192
    assert d.flops == 100.0
    assert d.memory_bandwidth_gbps == 1000
    assert d.node_id == "node0"
    assert d.socket_id == "socket0"


def test_link_transfer_time_positive():
    rng = np.random.default_rng(0)
    link = Link("a", "b", bandwidth_gbps=100.0, base_latency_us=5.0,
                jitter=Constant(0.0))
    t = link.sample_transfer_time(50.0, rng)
    assert t > 0.0


def test_link_transfer_time_scales_with_size():
    rng = np.random.default_rng(0)
    link = Link("a", "b", bandwidth_gbps=100.0, base_latency_us=0.0,
                jitter=Constant(0.0))
    t1 = link.sample_transfer_time(10.0, rng)
    t2 = link.sample_transfer_time(20.0, rng)
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
        [Device("gpu0", "gpu", 8192)],
        [],
    )
    link = topo.link("gpu0", "gpu0")
    rng = np.random.default_rng(0)
    assert link.sample_transfer_time(100.0, rng) == 0.0


def test_topology_infers_links_from_fabric_profiles():
    topo = Topology.from_yaml({
        "devices": [
            {"id": "n0_gpu0", "kind": "gpu", "memory_mb": 8192, "numa_node": 0},
            {"id": "n0_gpu1", "kind": "gpu", "memory_mb": 8192, "numa_node": 0},
            {"id": "n0_gpu2", "kind": "gpu", "memory_mb": 8192, "numa_node": 1},
            {"id": "n1_gpu0", "kind": "gpu", "memory_mb": 8192, "numa_node": 0},
        ],
        "fabric": {
            "same_socket": {
                "bandwidth_gbps": 1000,
                "base_latency_us": 1.0,
                "jitter": {"type": "constant", "value": 0.0},
            },
            "same_node": {
                "bandwidth_gbps": 300,
                "base_latency_us": 3.0,
                "jitter": {"type": "constant", "value": 0.0},
            },
            "cross_node": {
                "bandwidth_gbps": 100,
                "base_latency_us": 10.0,
                "jitter": {"type": "constant", "value": 0.0},
            },
        },
    })

    assert topo.link("n0_gpu0", "n0_gpu1").bandwidth_gbps == 1000
    assert topo.link("n0_gpu0", "n0_gpu2").bandwidth_gbps == 300
    assert topo.link("n0_gpu0", "n1_gpu0").bandwidth_gbps == 100


def test_explicit_link_overrides_fabric_profile():
    topo = Topology.from_yaml({
        "devices": [
            {"id": "n0_gpu0", "kind": "gpu", "memory_mb": 8192, "numa_node": 0},
            {"id": "n0_gpu1", "kind": "gpu", "memory_mb": 8192, "numa_node": 0},
        ],
        "links": [
            {
                "src": "n0_gpu0",
                "dst": "n0_gpu1",
                "bandwidth_gbps": 777,
                "base_latency_us": 2.0,
                "jitter": {"type": "constant", "value": 0.0},
            },
        ],
        "fabric": {
            "same_socket": {
                "bandwidth_gbps": 1000,
                "base_latency_us": 1.0,
                "jitter": {"type": "constant", "value": 0.0},
            },
        },
    })

    assert topo.link("n0_gpu0", "n0_gpu1").bandwidth_gbps == 777


def test_topology_parses_locality_penalties():
    topo = Topology.from_yaml({
        "devices": [
            {"id": "n0_gpu0", "kind": "gpu", "memory_mb": 8192, "node_id": "n0", "socket_id": "s0"},
            {"id": "n0_gpu1", "kind": "gpu", "memory_mb": 8192, "node_id": "n0", "socket_id": "s1"},
        ],
        "locality_penalties": {
            "same_node": {
                "compute_scale": 1.1,
                "memory_bandwidth_scale": 0.8,
                "memory_latency_us": 2.0,
                "transfer_scale": 1.2,
            },
        },
    })

    penalty = topo.locality_penalty(Locality.SAME_NODE)
    assert penalty.compute_scale == 1.1
    assert penalty.memory_bandwidth_scale == 0.8
    assert penalty.memory_latency_us == 2.0
    assert penalty.transfer_scale == 1.2
