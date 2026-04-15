"""Tests for the pipeline model."""

import numpy as np

from hops.config import parse_config
from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import GPipeScheduler, OneFOneBScheduler
from hops.core.types import Phase
from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import Locality, LocalityPenalty, Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant
from hops.metrics.collector import MetricsCollector

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


def test_optimizer_step_adds_time():
    """With optimizer enabled, batch should take longer than without."""
    rng = np.random.default_rng(0)
    num_stages = 2
    devices = [Device(f"gpu{i}", "gpu", 8192) for i in range(num_stages)]
    links = [
        Link("gpu0", "gpu1", 900, 0.0, Constant(0.0)),
        Link("gpu1", "gpu0", 900, 0.0, Constant(0.0)),
    ]
    topology = Topology(devices, links)
    compute_model = ComputeModel({i: Constant(5.0) for i in range(num_stages)})
    collector = MetricsCollector()
    engine = EventEngine()

    pipeline = Pipeline(
        [Stage(i, f"gpu{i}") for i in range(num_stages)],
        engine, topology, compute_model, GPipeScheduler(), collector,
        activation_size_mb=0.0, rng=rng,
        optimizer_latency=Constant(10.0), gradient_size_mb=50.0,
    )
    pipeline.start_batch(1)
    engine.run(stop_condition=lambda: pipeline.batch_complete)

    # Verify optimizer compute and all-reduce transfer records exist
    opt_computes = [r for r in collector.computes if r.phase == Phase.OPTIMIZER]
    opt_transfers = [t for t in collector.transfers if t.phase == Phase.OPTIMIZER]
    assert len(opt_computes) == num_stages  # one per device
    assert len(opt_transfers) == 2 * (num_stages - 1)  # bidirectional adjacent pairs

    # Total time should be > fwd+bwd alone (30ms for 1MB, 2 stages, constant 5ms)
    makespan = collector.makespan()
    assert makespan > 30.0


def test_optimizer_disabled_by_default():
    """Without optimizer config, batch completes normally."""
    engine, pipeline, collector = make_test_pipeline(
        GPipeScheduler(), num_stages=2, compute_time=10.0, seed=0)
    pipeline.start_batch(1)
    engine.run(stop_condition=lambda: pipeline.batch_complete)

    opt_computes = [r for r in collector.computes if r.phase == Phase.OPTIMIZER]
    assert len(opt_computes) == 0
    assert pipeline.batch_complete


def test_derived_stage_latency_uses_device_capabilities():
    rng = np.random.default_rng(0)
    topology = Topology([
        Device(
            "gpu0",
            "gpu",
            8192,
            flops=100.0,
            memory_bandwidth_gbps=400.0,
        )
    ], [])
    config = parse_config({
        "simulation": {"batches": 1, "microbatches": 1, "seed": 0},
        "pipeline": {
            "schedule": "gpipe",
            "precision": "fp32",
            "activation_mb": 0.0,
            "backward_factor": 2.0,
            "stages": [{
                "device": "gpu0",
                "weights_mb": 0.0,
                "compute": {
                    "mode": "analytical",
                    "tflop": 10.0,
                    "memory_mb": 100.0,
                    "efficiency": {"compute": 1.0, "memory": 1.0},
                },
            }],
        },
        "hardware": {
            "devices": [{"id": "gpu0", "gpu": "a100", "node": "node0", "socket": 0}],
            "interconnect": {"same_node": "nvlink", "cross_node": "infiniband"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    })
    model = ComputeModel.from_pipeline_config(config.pipeline, topology=topology)

    sample = model.sample(0, Phase.FORWARD, rng)
    expected_compute = 1000.0 * 10.0 / 100.0
    expected_memory = (100.0 * 8.0) / 400.0
    assert abs(sample - (expected_compute + expected_memory)) < 1e-9


def test_per_device_utilization_aggregates_compute_on_shared_device():
    collector = MetricsCollector()
    collector.record_compute(0, 0, Phase.FORWARD, "gpu0", 0.0, 2.0)
    collector.record_compute(1, 0, Phase.BACKWARD, "gpu0", 2.0, 5.0)
    collector.record_compute(2, 0, Phase.FORWARD, "gpu1", 1.0, 4.0)

    util = collector.per_device_utilization()

    assert util["gpu0"] == 1.0
    assert abs(util["gpu1"] - 0.6) < 1e-9


def test_link_utilization_and_contention_stats_are_reported():
    collector = MetricsCollector()
    collector.record_transfer(0, Phase.FORWARD, "gpu0", "gpu1", 0.0, 2.0)
    collector.record_transfer(1, Phase.FORWARD, "gpu0", "gpu1", 1.0, 3.0)
    collector.record_transfer(2, Phase.BACKWARD, "gpu1", "gpu0", 0.0, 1.0)

    link_util = collector.per_link_transfer_utilization()
    contention = collector.transfer_contention_stats()

    assert link_util["gpu0->gpu1"] == 1.0
    assert abs(link_util["gpu1->gpu0"] - (1.0 / 3.0)) < 1e-9
    assert contention["global_peak_concurrency"] == 2.0
    assert contention["per_link"]["gpu0->gpu1"]["contended_transfer_fraction"] == 1.0


def test_derived_latency_applies_memory_locality_penalty():
    rng = np.random.default_rng(0)
    topology = Topology(
        [
            Device(
                "gpu0",
                "gpu",
                8192,
                flops=100.0,
                memory_bandwidth_gbps=400.0,
                node_id="node0",
                socket_id="socket0",
            )
        ],
        [],
        locality_penalties={
            Locality.SAME_NODE: LocalityPenalty(
                compute_scale=1.2,
                memory_bandwidth_scale=0.5,
                memory_latency_us=1000.0,
            ),
        },
    )
    config = parse_config({
        "simulation": {"batches": 1, "microbatches": 1, "seed": 0},
        "pipeline": {
            "schedule": "gpipe",
            "precision": "fp32",
            "activation_mb": 0.0,
            "backward_factor": 2.0,
            "stages": [{
                "device": "gpu0",
                "weights_mb": 0.0,
                "compute": {
                    "mode": "analytical",
                    "tflop": 10.0,
                    "memory_mb": 100.0,
                    "efficiency": {"compute": 1.0, "memory": 1.0},
                },
                "memory_placement": {
                    "kind": "socket",
                    "node": "node0",
                    "socket": "socket1",
                },
            }],
        },
        "hardware": {
            "devices": [{"id": "gpu0", "gpu": "a100", "node": "node0", "socket": 0}],
            "interconnect": {"same_node": "nvlink", "cross_node": "infiniband"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    })
    model = ComputeModel.from_pipeline_config(config.pipeline, topology=topology)

    sample = model.sample(0, Phase.FORWARD, rng)
    expected_compute = (1000.0 * 10.0 / 100.0) * 1.2
    expected_memory = (100.0 * 8.0) / (400.0 * 0.5) + 1.0
    assert abs(sample - (expected_compute + expected_memory)) < 1e-9
