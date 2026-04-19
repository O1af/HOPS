"""Tests for hardware topology."""

import numpy as np

from hops.core.timing import TimingModel
from hops.hardware.device import Device, numa_from_socket
from hops.hardware.network import Link
from hops.hardware.topology import LinkProfile, Locality, LocalityPenalty, Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant


def test_numa_from_socket_extracts_digits():
    assert numa_from_socket("socket0") == 0
    assert numa_from_socket("s12") == 12


def test_link_transfer_time_positive():
    rng = np.random.default_rng(0)
    topology = Topology(
        [Device("a", "gpu", 8192), Device("b", "gpu", 8192)],
        [Link("a", "b", bandwidth_gbps=100.0, base_latency_us=5.0, jitter=Constant(0.0))],
    )
    timing = TimingModel(topology, ComputeModel({}), rng)
    _, end_time, _ = timing.reserve_transfer(now=0.0, src_device="a", dst_device="b", size_mb=50.0)
    assert end_time > 0.0


def test_link_transfer_time_scales_with_size():
    rng = np.random.default_rng(0)
    topology = Topology(
        [Device("a", "gpu", 8192), Device("b", "gpu", 8192)],
        [Link("a", "b", bandwidth_gbps=100.0, base_latency_us=0.0, jitter=Constant(0.0))],
    )
    timing = TimingModel(topology, ComputeModel({}), rng)
    _, t1, tid1 = timing.reserve_transfer(now=0.0, src_device="a", dst_device="b", size_mb=10.0)
    timing.release_transfer("a", "b", tid1, t1)
    _, t2, _ = timing.reserve_transfer(now=0.0, src_device="a", dst_device="b", size_mb=20.0)
    assert abs(t2 - 2 * t1) < 1e-10


def test_topology_explicit_links_are_available():
    topo = Topology(
        [
            Device("gpu0", "gpu", 8192, flops=100.0, memory_bandwidth_gbps=1000.0),
            Device("gpu1", "gpu", 8192, flops=100.0, memory_bandwidth_gbps=1000.0),
        ],
        [Link("gpu0", "gpu1", bandwidth_gbps=100.0, base_latency_us=1.0, jitter=Constant(0.0))],
    )
    assert topo.device("gpu0").kind == "gpu"
    assert topo.link("gpu0", "gpu1").bandwidth_gbps == 100


def test_same_device_link_is_free():
    topo = Topology(
        [Device("gpu0", "gpu", 8192)],
        [],
    )
    link = topo.link("gpu0", "gpu0")
    assert link.base_latency_us == 0.0
    assert link.bandwidth_gbps == float("inf")


def test_topology_infers_links_from_fabric_profiles():
    topo = Topology(
        [
            Device("n0_gpu0", "gpu", 8192, node_id="n0", socket_id="s0"),
            Device("n0_gpu1", "gpu", 8192, node_id="n0", socket_id="s0"),
            Device("n0_gpu2", "gpu", 8192, node_id="n0", socket_id="s1"),
            Device("n1_gpu0", "gpu", 8192, node_id="n1", socket_id="s0"),
        ],
        [],
        link_profiles={
            Locality.SAME_SOCKET: LinkProfile(1000.0, 1.0, Constant(0.0)),
            Locality.SAME_NODE: LinkProfile(300.0, 3.0, Constant(0.0)),
            Locality.CROSS_NODE: LinkProfile(100.0, 10.0, Constant(0.0)),
        },
    )

    assert topo.link("n0_gpu0", "n0_gpu1").bandwidth_gbps == 1000
    assert topo.link("n0_gpu0", "n0_gpu2").bandwidth_gbps == 300
    assert topo.link("n0_gpu0", "n1_gpu0").bandwidth_gbps == 100


def test_explicit_link_overrides_fabric_profile():
    topo = Topology(
        [
            Device("n0_gpu0", "gpu", 8192, node_id="n0", socket_id="s0"),
            Device("n0_gpu1", "gpu", 8192, node_id="n0", socket_id="s0"),
        ],
        [Link("n0_gpu0", "n0_gpu1", 777.0, 2.0, Constant(0.0))],
        link_profiles={
            Locality.SAME_SOCKET: LinkProfile(1000.0, 1.0, Constant(0.0)),
        },
    )

    assert topo.link("n0_gpu0", "n0_gpu1").bandwidth_gbps == 777


def test_topology_parses_locality_penalties():
    topo = Topology(
        [
            Device("n0_gpu0", "gpu", 8192, node_id="n0", socket_id="s0"),
            Device("n0_gpu1", "gpu", 8192, node_id="n0", socket_id="s1"),
        ],
        [],
        locality_penalties={
            Locality.SAME_NODE: LocalityPenalty(
                compute_scale=1.1,
                memory_bandwidth_scale=0.8,
                memory_latency_us=2.0,
                transfer_scale=1.2,
            ),
        },
    )

    penalty = topo.locality_penalty(Locality.SAME_NODE)
    assert penalty.compute_scale == 1.1
    assert penalty.memory_bandwidth_scale == 0.8
    assert penalty.memory_latency_us == 2.0
    assert penalty.transfer_scale == 1.2
