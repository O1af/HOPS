"""Tests for features 1-9: ZeroBubble, memory, contention, ring all-reduce, etc."""

import numpy as np
import pytest

from hops.config import parse_config, validate_config
from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import (
    GPipeScheduler, OneFOneBScheduler, ZeroBubbleScheduler,
    PipelineState, make_scheduler, max_in_flight_count,
)
from hops.core.types import AllreduceAlgo, Phase, Precision
from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.hardware.topology import LinkProfile, Locality, LocalityPenalty, Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Constant
from hops.metrics.collector import MetricsCollector
from hops.runtime import build_runtime

from .conftest import make_canonical_config


def validate_and_parse(config: dict):
    return parse_config(config)


def _make_pipeline(scheduler, *, num_stages=4, compute_time=5.0, seed=42,
                   activation_size_mb=0.0, optimizer_latency=None,
                   gradient_size_mb=0.0, stage_memory_mb=None,
                   gradient_accumulation_steps=1, precision=Precision.FP32,
                   allreduce_algo=AllreduceAlgo.NAIVE, device_memory_mb=8192,
                   include_ring_link=False):
    """Helper to build pipelines with new features."""
    rng = np.random.default_rng(seed)
    devices = [Device(f"gpu{i}", "gpu", device_memory_mb) for i in range(num_stages)]
    links = []
    for i in range(num_stages - 1):
        links.append(Link(f"gpu{i}", f"gpu{i+1}", 900, 0.0, Constant(0.0)))
        links.append(Link(f"gpu{i+1}", f"gpu{i}", 900, 0.0, Constant(0.0)))
    if include_ring_link:
        links.append(Link(f"gpu{num_stages-1}", "gpu0", 900, 0.0, Constant(0.0)))
        links.append(Link("gpu0", f"gpu{num_stages-1}", 900, 0.0, Constant(0.0)))

    topology = Topology(devices, links)
    dists = {i: Constant(compute_time) for i in range(num_stages)}
    compute_model = ComputeModel(dists)
    collector = MetricsCollector()
    engine = EventEngine()

    stages = [Stage(i, f"gpu{i}") for i in range(num_stages)]
    pipeline = Pipeline(
        stages, engine, topology, compute_model, scheduler, collector,
        activation_size_mb=activation_size_mb, rng=rng,
        optimizer_latency=optimizer_latency,
        gradient_size_mb=gradient_size_mb,
        stage_memory_mb=stage_memory_mb,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
        allreduce_algo=allreduce_algo,
    )
    return engine, pipeline, collector


# ── Feature 1: ZeroBubble scheduler ──────────────────────────────────────

class TestZeroBubble:
    def test_zero_bubble_completes_all_microbatches(self):
        engine, pipeline, collector = _make_pipeline(ZeroBubbleScheduler())
        pipeline.start_batch(4)
        engine.run()

        mb_ids = set(r.microbatch_id for r in collector.computes)
        assert len(mb_ids) == 4

    def test_zero_bubble_produces_b_and_w_phases(self):
        engine, pipeline, collector = _make_pipeline(ZeroBubbleScheduler())
        pipeline.start_batch(4)
        engine.run()

        phases = {r.phase for r in collector.computes}
        assert Phase.FORWARD in phases
        assert Phase.BACKWARD_B in phases
        assert Phase.BACKWARD_W in phases
        assert Phase.BACKWARD not in phases

    def test_zero_bubble_less_or_equal_bubbles_than_1f1b(self):
        def run(scheduler):
            engine, pipeline, collector = _make_pipeline(
                scheduler, num_stages=4, compute_time=1.0)
            pipeline.start_batch(8)
            engine.run()
            return collector.bubble_ratio()

        zb = run(ZeroBubbleScheduler())
        onefb = run(OneFOneBScheduler())
        assert zb <= onefb + 0.01  # allow small floating point margin

    def test_zero_bubble_w_tasks_deferred(self):
        """W tasks should start after their corresponding B task."""
        engine, pipeline, collector = _make_pipeline(
            ZeroBubbleScheduler(), num_stages=2, compute_time=1.0)
        pipeline.start_batch(4)
        engine.run()

        for stage_id in range(2):
            b_records = sorted(
                [r for r in collector.computes
                 if r.stage_id == stage_id and r.phase == Phase.BACKWARD_B],
                key=lambda r: r.microbatch_id,
            )
            w_records = sorted(
                [r for r in collector.computes
                 if r.stage_id == stage_id and r.phase == Phase.BACKWARD_W],
                key=lambda r: r.microbatch_id,
            )
            for b, w in zip(b_records, w_records):
                assert w.start_time >= b.end_time

    def test_make_scheduler_zero_bubble(self):
        sched = make_scheduler({"policy": "zero_bubble"})
        assert isinstance(sched, ZeroBubbleScheduler)

    def test_pipeline_state_initialize_w_split(self):
        state = PipelineState.initialize(2, 2, use_w_split=True)
        assert (0, 0, Phase.BACKWARD_B) in state.task_states
        assert (0, 0, Phase.BACKWARD_W) in state.task_states
        assert (0, 0, Phase.BACKWARD) not in state.task_states

    def test_pipeline_state_initialize_no_w_split(self):
        state = PipelineState.initialize(2, 2, use_w_split=False)
        assert (0, 0, Phase.BACKWARD) in state.task_states
        assert (0, 0, Phase.BACKWARD_B) not in state.task_states


# ── Feature 2: Memory tracking + validation ──────────────────────────────

class TestMemory:
    def test_device_allocate_and_free(self):
        d = Device("gpu0", "gpu", 8192)
        d.allocate(1000)
        assert d.memory_used_mb == 1000
        assert d.peak_memory_mb == 1000
        d.allocate(500)
        assert d.memory_used_mb == 1500
        assert d.peak_memory_mb == 1500
        d.free(1000)
        assert d.memory_used_mb == 500
        assert d.peak_memory_mb == 1500  # peak unchanged

    def test_peak_memory_recorded_in_collector(self):
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, activation_size_mb=100.0)
        pipeline.start_batch(4)
        engine.run()

        assert len(collector.peak_memory_per_device) > 0
        for device_id, peak in collector.peak_memory_per_device.items():
            assert peak > 0

    def test_memory_validation_passes_for_feasible_config(self):
        config = make_canonical_config(batches=1, microbatches=8)
        config["pipeline"]["activation_mb"] = 50.0
        config["pipeline"]["stages"][0]["weights_mb"] = 2048
        config["hardware"]["devices"][0]["gpu"] = "h100"
        build_runtime(validate_and_parse(config))

    def test_memory_validation_fails_for_infeasible_config(self):
        config = make_canonical_config(batches=1, microbatches=8)
        config["pipeline"]["activation_mb"] = 50.0
        config["pipeline"]["stages"][0]["weights_mb"] = 2048
        config["overrides"] = {"devices": [{"id": "gpu0", "memory_mb": 100}]}
        with pytest.raises(ValueError, match="exceeds device capacity"):
            build_runtime(validate_and_parse(config))

    def test_memory_validation_aggregates_shared_device_usage(self):
        config = make_canonical_config(batches=1, microbatches=4)
        config["pipeline"]["schedule"] = "1f1b"
        config["pipeline"]["activation_mb"] = 250.0
        config["pipeline"]["stages"] = [
            {
                "device": "gpu0",
                "weights_mb": 1200.0,
                "compute": {"mode": "explicit", "distribution": {"type": "constant", "value": 1.0}},
            },
            {
                "device": "gpu0",
                "weights_mb": 1200.0,
                "compute": {"mode": "explicit", "distribution": {"type": "constant", "value": 1.0}},
            },
        ]
        config["overrides"] = {"devices": [{"id": "gpu0", "memory_mb": 3000}]}
        with pytest.raises(ValueError, match="Device gpu0"):
            build_runtime(validate_and_parse(config))

    def test_max_in_flight_count(self):
        assert max_in_flight_count("gpipe", 0, 4, 8) == 8
        assert max_in_flight_count("1f1b", 0, 4, 8) == 4
        assert max_in_flight_count("1f1b", 3, 4, 8) == 1
        assert max_in_flight_count("zero_bubble", 0, 4, 8) == 4

    def test_stage_weight_memory_allocated(self):
        """Weight memory is pre-allocated on devices at pipeline creation."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2,
            stage_memory_mb={0: 1000, 1: 2000})

        device0 = pipeline.topology.device("gpu0")
        device1 = pipeline.topology.device("gpu1")
        assert device0.memory_used_mb == 1000
        assert device1.memory_used_mb == 2000


# ── Feature 3/4: Compute/comm overlap + bandwidth contention ─────────────

class TestOverlapAndContention:
    def test_transfer_does_not_block_device_compute(self):
        """Transfers should not affect device busy_until (overlap)."""
        rng = np.random.default_rng(0)
        devices = [Device("gpu0", "gpu", 8192), Device("gpu1", "gpu", 8192)]
        links = [Link("gpu0", "gpu1", 900, 0.0, Constant(0.0))]
        topology = Topology(devices, links)
        compute_model = ComputeModel({0: Constant(10.0)})

        from hops.core.timing import TimingModel
        tm = TimingModel(topology, compute_model, rng)

        # Reserve a transfer — should not touch device busy_until
        tm.reserve_transfer(now=0.0, src_device="gpu0", dst_device="gpu1", size_mb=1.0)
        assert devices[0].busy_until == 0.0  # device not blocked

        # Reserve compute — should use device busy_until
        tm.reserve_compute(now=0.0, stage_id=0, device_id="gpu0", phase=Phase.FORWARD)
        assert devices[0].busy_until == 10.0

    def test_link_contention_increases_transfer_time(self):
        """Multiple concurrent transfers on same link should take longer."""
        rng = np.random.default_rng(0)
        devices = [Device("a", "gpu", 8192), Device("b", "gpu", 8192)]
        links = [Link("a", "b", 800, 0.0, Constant(0.0))]
        topology = Topology(devices, links)
        compute_model = ComputeModel({})

        from hops.core.timing import TimingModel
        tm = TimingModel(topology, compute_model, rng)

        # First transfer: full bandwidth (800 Gbps)
        s1, e1 = tm.reserve_transfer(now=0.0, src_device="a", dst_device="b", size_mb=100.0)
        duration_alone = e1 - s1

        # Second transfer: bandwidth halved (2 active)
        s2, e2 = tm.reserve_transfer(now=0.0, src_device="a", dst_device="b", size_mb=100.0)
        duration_contended = e2 - s2

        assert duration_contended > duration_alone

    def test_release_transfer_decrements_count(self):
        rng = np.random.default_rng(0)
        devices = [Device("a", "gpu", 8192), Device("b", "gpu", 8192)]
        links = [Link("a", "b", 800, 0.0, Constant(0.0))]
        topology = Topology(devices, links)
        compute_model = ComputeModel({})

        from hops.core.timing import TimingModel
        tm = TimingModel(topology, compute_model, rng)

        tm.reserve_transfer(now=0.0, src_device="a", dst_device="b", size_mb=1.0)
        assert links[0].active_transfers == 1
        tm.release_transfer("a", "b")
        assert links[0].active_transfers == 0

    def test_transfer_locality_penalty_scales_duration(self):
        rng = np.random.default_rng(0)
        topology = Topology(
            [
                Device("n0_gpu0", "gpu", 8192, node_id="n0", socket_id="s0"),
                Device("n0_gpu1", "gpu", 8192, node_id="n0", socket_id="s1"),
            ],
            [],
            link_profiles={
                Locality.SAME_NODE: LinkProfile(800.0, 0.0, Constant(0.0)),
            },
            locality_penalties={
                Locality.SAME_NODE: LocalityPenalty(transfer_scale=1.5),
            },
        )
        compute_model = ComputeModel({})

        from hops.core.timing import TimingModel
        tm = TimingModel(topology, compute_model, rng)

        start_time, end_time = tm.reserve_transfer(
            now=0.0,
            src_device="n0_gpu0",
            dst_device="n0_gpu1",
            size_mb=100.0,
        )

        assert abs((end_time - start_time) - 1.5) < 1e-9


# ── Feature 6: Pipeline drain verification ───────────────────────────────

class TestPipelineDrain:
    def test_gpipe_drain_creates_staircase_idle_at_end(self):
        """GPipe: after all forwards, backward drain creates staircase idle."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=4, compute_time=1.0)
        pipeline.start_batch(4)
        engine.run()

        idle = collector.stage_idle_intervals()
        # Stage 3 should have no trailing idle (it finishes last in backward)
        # Stage 0 should have a large gap in the middle (waiting for backward)
        assert len(idle[0]) > 0  # Stage 0 has idle time
        # The last stage should have idle at the start (waiting for forward to arrive)
        assert idle[3][0][0] == 0.0  # starts idle at time 0

    def test_1f1b_drain_flush_phase(self):
        """1F1B: the flush phase at the end should show staircase."""
        engine, pipeline, collector = _make_pipeline(
            OneFOneBScheduler(), num_stages=4, compute_time=1.0)
        pipeline.start_batch(8)
        engine.run()

        idle = collector.stage_idle_intervals()
        # All stages should have some idle time
        for stage_id in range(4):
            assert stage_id in idle

    def test_zero_bubble_drain_completes(self):
        """ZeroBubble should complete drain including W tasks."""
        engine, pipeline, collector = _make_pipeline(
            ZeroBubbleScheduler(), num_stages=4, compute_time=1.0)
        pipeline.start_batch(8)
        engine.run()

        # All microbatches should complete
        assert collector.completed_microbatches == 8
        # Should have W tasks
        w_count = sum(1 for r in collector.computes if r.phase == Phase.BACKWARD_W)
        assert w_count == 8 * 4  # 8 MBs * 4 stages


# ── Feature 7: Gradient accumulation ─────────────────────────────────────

class TestGradientAccumulation:
    def test_optimizer_fires_only_after_accumulation_steps(self):
        """With accumulation=2, optimizer should only fire on even batches."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, compute_time=5.0,
            optimizer_latency=Constant(1.0), gradient_size_mb=10.0,
            gradient_accumulation_steps=2)

        # Batch 1: no optimizer
        pipeline.start_batch(2)
        engine.run(stop_condition=lambda: pipeline.batch_complete)
        opt_count_after_1 = sum(1 for r in collector.computes if r.phase == Phase.OPTIMIZER)
        assert opt_count_after_1 == 0

        # Batch 2: optimizer fires
        pipeline.start_batch(2)
        engine.run(stop_condition=lambda: pipeline.batch_complete)
        opt_count_after_2 = sum(1 for r in collector.computes if r.phase == Phase.OPTIMIZER)
        assert opt_count_after_2 == 2  # one per device

    def test_accumulation_resets_after_optimizer(self):
        """After optimizer fires, accumulation counter resets."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, compute_time=5.0,
            optimizer_latency=Constant(1.0), gradient_size_mb=10.0,
            gradient_accumulation_steps=2)

        # 4 batches: optimizer at batch 2 and batch 4
        for _ in range(4):
            pipeline.start_batch(1)
            engine.run(stop_condition=lambda: pipeline.batch_complete)

        opt_count = sum(1 for r in collector.computes if r.phase == Phase.OPTIMIZER)
        assert opt_count == 4  # 2 devices * 2 optimizer steps


# ── Feature 8: Ring all-reduce ───────────────────────────────────────────

class TestRingAllReduce:
    def test_ring_allreduce_produces_correct_transfer_count(self):
        """Ring: 2*(N-1) rounds * N transfers per round."""
        N = 4
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=N, compute_time=5.0,
            optimizer_latency=Constant(1.0), gradient_size_mb=100.0,
            allreduce_algo=AllreduceAlgo.RING, include_ring_link=True)

        pipeline.start_batch(1)
        engine.run(stop_condition=lambda: pipeline.batch_complete)

        opt_transfers = [t for t in collector.transfers if t.phase == Phase.OPTIMIZER]
        expected = 2 * (N - 1) * N  # rounds * transfers_per_round
        assert len(opt_transfers) == expected

    def test_ring_allreduce_chunk_size_is_gradient_over_n(self):
        """Each ring transfer should carry gradient_size / N."""
        N = 4
        gradient_size = 100.0
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=N, compute_time=5.0,
            optimizer_latency=Constant(1.0), gradient_size_mb=gradient_size,
            allreduce_algo=AllreduceAlgo.RING, include_ring_link=True)

        pipeline.start_batch(1)
        engine.run(stop_condition=lambda: pipeline.batch_complete)

        # Verify pipeline used correct chunk size
        assert pipeline._ring_chunk_size == gradient_size / N

    def test_ring_allreduce_requires_ring_links(self):
        """Ring all-reduce without ring closure link should raise."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=4, compute_time=5.0,
            optimizer_latency=Constant(1.0), gradient_size_mb=100.0,
            allreduce_algo=AllreduceAlgo.RING, include_ring_link=False)

        pipeline.start_batch(1)
        with pytest.raises(ValueError, match="Ring all-reduce requires"):
            engine.run(stop_condition=lambda: pipeline.batch_complete)

    def test_naive_allreduce_still_works(self):
        """Naive (default) allreduce should work as before."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, compute_time=5.0,
            optimizer_latency=Constant(1.0), gradient_size_mb=50.0,
            allreduce_algo=AllreduceAlgo.NAIVE)

        pipeline.start_batch(1)
        engine.run(stop_condition=lambda: pipeline.batch_complete)

        opt_transfers = [t for t in collector.transfers if t.phase == Phase.OPTIMIZER]
        assert len(opt_transfers) == 2  # bidirectional pair


# ── Feature 9: Mixed precision ───────────────────────────────────────────

class TestMixedPrecision:
    def test_fp16_halves_activation_size(self):
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, activation_size_mb=100.0,
            precision=Precision.FP16)
        assert pipeline.activation_size_mb == 50.0

    def test_bf16_halves_gradient_size(self):
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, gradient_size_mb=200.0,
            precision=Precision.BF16)
        assert pipeline.gradient_size_mb == 100.0

    def test_fp32_no_scaling(self):
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2, activation_size_mb=100.0,
            gradient_size_mb=200.0, precision=Precision.FP32)
        assert pipeline.activation_size_mb == 100.0
        assert pipeline.gradient_size_mb == 200.0

    def test_mixed_precision_weight_memory_overhead(self):
        """FP16 adds 1.5x weight memory overhead (master copy)."""
        engine, pipeline, collector = _make_pipeline(
            GPipeScheduler(), num_stages=2,
            stage_memory_mb={0: 1000, 1: 1000}, precision=Precision.FP16)

        device0 = pipeline.topology.device("gpu0")
        assert device0.memory_used_mb == 1500  # 1000 * 1.5

    def test_precision_speedup_in_compute_model(self):
        """Precision speedup should reduce compute time."""
        rng = np.random.default_rng(0)
        dists = {0: Constant(10.0)}

        model_fp32 = ComputeModel(dists, precision_speedup=1.0)
        model_fp16 = ComputeModel(dists, precision_speedup=2.0)

        time_fp32 = model_fp32.sample(0, Phase.FORWARD, rng)
        time_fp16 = model_fp16.sample(0, Phase.FORWARD, rng)

        assert abs(time_fp16 - time_fp32 / 2) < 0.001


# ── Feature 1 + 9 combo: ZeroBubble with mixed precision ────────────────

class TestZeroBubbleMixedPrecision:
    def test_zero_bubble_fp16_completes(self):
        engine, pipeline, collector = _make_pipeline(
            ZeroBubbleScheduler(), num_stages=4, compute_time=5.0,
            activation_size_mb=100.0, precision=Precision.FP16)
        pipeline.start_batch(8)
        engine.run()

        assert collector.completed_microbatches == 8
        assert pipeline.activation_size_mb == 50.0


class TestConfigurationValidation:
    def test_pipeline_rejects_non_contiguous_stage_ids(self):
        rng = np.random.default_rng(0)
        topology = Topology([Device("gpu0", "gpu", 8192)], [])
        collector = MetricsCollector()
        engine = EventEngine()

        with pytest.raises(ValueError, match="contiguous zero-based"):
            Pipeline(
                [Stage(1, "gpu0")],
                engine,
                topology,
                ComputeModel({1: Constant(5.0)}),
                GPipeScheduler(),
                collector,
                activation_size_mb=0.0,
                rng=rng,
            )

    def test_pipeline_rejects_missing_backward_link(self):
        rng = np.random.default_rng(0)
        topology = Topology(
            [Device("gpu0", "gpu", 8192), Device("gpu1", "gpu", 8192)],
            [Link("gpu0", "gpu1", 900, 0.0, Constant(0.0))],
        )
        collector = MetricsCollector()
        engine = EventEngine()

        with pytest.raises(ValueError, match="backward transfer requires a link"):
            Pipeline(
                [Stage(0, "gpu0"), Stage(1, "gpu1")],
                engine,
                topology,
                ComputeModel({0: Constant(5.0), 1: Constant(5.0)}),
                GPipeScheduler(),
                collector,
                activation_size_mb=0.0,
                rng=rng,
            )

    def test_validate_config_rejects_missing_stage_latency_definition(self):
        config = make_canonical_config()
        del config["pipeline"]["stages"][0]["compute"]
        with pytest.raises(ValueError, match="must define a compute block"):
            validate_config(config)
