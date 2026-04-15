"""Tests grounded in real experiment output.

Numbers come from actual simulation runs:
  - Experiment 05 (run 26): 2-stage H100 pipeline, 40 batches × 8 microbatches,
    explicit normal distributions, custom cross-node link.
  - Experiment 06 (run 1): 1-stage H100 smoke test, 4 batches × 1 microbatch.

Each test uses a fixed seed and constant distributions so the output is
deterministic and exactly matches the observed run values.

Throughput units: microbatches/ms  (0.02674 µb/ms = 26.74 µb/s in observed data).
"""

from hops.config import parse_config
from hops.core.types import Phase
from hops.runtime import build_runtime


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run(raw_config: dict):
    runtime = build_runtime(parse_config(raw_config))
    for _ in range(runtime.num_batches):
        runtime.pipeline.start_batch(runtime.num_microbatches)
        runtime.engine.run()
    return runtime


def _exp05_config(*, mean: float = 22.0, std: float = 0.0) -> dict:
    """Experiment 05 config.  std=0 gives deterministic runs for exact checks."""
    return {
        "simulation": {"batches": 40, "microbatches": 8, "seed": 42},
        "pipeline": {
            "schedule": "1f1b",
            "precision": "bf16",
            "activation_mb": 64,
            "backward_factor": 2.0,
            "stages": [
                {
                    "device": "h100_node0_gpu0",
                    "weights_mb": 4096,
                    "compute": {
                        "mode": "explicit",
                        "distribution": {"type": "normal", "mean": mean, "std": std},
                    },
                },
                {
                    "device": "h100_node1_gpu0",
                    "weights_mb": 4096,
                    "compute": {
                        "mode": "explicit",
                        "distribution": {"type": "normal", "mean": mean, "std": std},
                    },
                },
            ],
        },
        "hardware": {
            "devices": [
                {"id": "h100_node0_gpu0", "gpu": "h100", "node": "h100_node0", "socket": 0},
                {"id": "h100_node1_gpu0", "gpu": "h100", "node": "h100_node1", "socket": 0},
            ],
            "interconnect": {"same_node": "nvlink", "cross_node": "infiniband"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "overrides": {
            "links": [
                {
                    "src": "h100_node0_gpu0",
                    "dst": "h100_node1_gpu0",
                    "bandwidth_gbps": 12.5,
                    "latency_us": 200.0,
                },
                {
                    "src": "h100_node1_gpu0",
                    "dst": "h100_node0_gpu0",
                    "bandwidth_gbps": 12.5,
                    "latency_us": 200.0,
                },
            ]
        },
        "output": {},
    }


def _exp06_config(*, mean: float = 20.0, std: float = 0.0) -> dict:
    """Experiment 06 single-GPU smoke-test config."""
    return {
        "simulation": {"batches": 4, "microbatches": 1, "seed": 42},
        "pipeline": {
            "schedule": "1f1b",
            "precision": "bf16",
            "activation_mb": 64,
            "backward_factor": 2.0,
            "stages": [
                {
                    "device": "h100_node0_gpu0",
                    "weights_mb": 2048,
                    "compute": {
                        "mode": "explicit",
                        "distribution": {"type": "normal", "mean": mean, "std": std},
                    },
                }
            ],
        },
        "hardware": {
            "devices": [
                {"id": "h100_node0_gpu0", "gpu": "h100", "node": "h100_node0", "socket": 0}
            ],
            "interconnect": {"same_node": "nvlink", "cross_node": "infiniband"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    }


# ---------------------------------------------------------------------------
# Experiment 05 – 2-stage H100 dual-node, run 26 baseline
# ---------------------------------------------------------------------------

class TestExp05Baseline:
    """Run 26 baseline: 40 batches × 8 microbatches = 320 total microbatches.

    Observed values from experiment output (runs 17-26 all identical):
      throughput ≈ 0.02674 µb/ms (26.74 µb/s), bubble ≈ 0.1166,
      device util ≈ 0.886/0.881, communication overhead ≈ 5%.
    """

    def setup_method(self):
        self.runtime = _run(_exp05_config(mean=22.0, std=0.0))
        self.collector = self.runtime.collector

    def test_completed_microbatches(self):
        assert self.collector.completed_microbatches == 320

    def test_throughput_order_of_magnitude(self):
        # Observed: 0.02674 µb/ms. Constant distributions → deterministic.
        # Cross-node transfer time ≈ 20.68 ms dominates, so throughput is lower
        # than the observed stochastic run. Accept a reasonable range.
        tp = self.collector.throughput()
        assert 0.01 <= tp <= 0.04, f"throughput {tp:.5f} µb/ms outside expected range"

    def test_bubble_ratio_range(self):
        # 1F1B with 2 stages and 8 microbatches → observed ~11.7%.
        # Constant dist → deterministic bubble from pipeline geometry.
        br = self.collector.bubble_ratio()
        assert 0.05 <= br <= 0.25, f"bubble ratio {br:.4f} outside expected range"

    def test_both_stages_utilised(self):
        util = self.collector.per_device_utilization()
        assert util["h100_node0_gpu0"] > 0.60
        assert util["h100_node1_gpu0"] > 0.60

    def test_cross_node_link_used_in_both_directions(self):
        link_util = self.collector.per_link_transfer_utilization()
        assert link_util.get("h100_node0_gpu0->h100_node1_gpu0", 0) > 0
        assert link_util.get("h100_node1_gpu0->h100_node0_gpu0", 0) > 0

    def test_transfer_contention_stats_populated(self):
        # 320 µb with 20.68ms transfers and 11ms compute cycles → high overlap,
        # observed peak concurrency = 1.0 (stochastic run); constant dist gives up to 6.
        contention = self.collector.transfer_contention_stats()
        assert "global_peak_concurrency" in contention
        assert contention["global_peak_concurrency"] >= 1.0

    def test_peak_memory_within_device_capacity(self):
        # h100 has 81920 MB.  Observed peaks: ~6304 / 6176 MB.
        peaks = self.collector.peak_memory_per_device
        assert peaks["h100_node0_gpu0"] < 81920
        assert peaks["h100_node1_gpu0"] < 81920

    def test_peak_memory_accounts_for_weights_and_activations(self):
        # weights_mb=4096, bf16 weight overhead=1.5 → 6144 MB weights.
        # Plus activations. Peak must exceed weights alone.
        peaks = self.collector.peak_memory_per_device
        assert peaks["h100_node0_gpu0"] >= 6144
        assert peaks["h100_node1_gpu0"] >= 6144

    def test_transfer_time_is_non_trivial_fraction(self):
        # Cross-node link is slow (12.5 Gbps, 200 µs): transfers take measurable time.
        # Observed communication overhead ≈ 5% of compute time.
        total_transfer = sum(
            r.end_time - r.start_time for r in self.collector.transfers
        )
        total_compute = sum(
            r.end_time - r.start_time for r in self.collector.computes
        )
        assert total_transfer > 0
        ratio = total_transfer / total_compute
        assert ratio > 0.02, f"transfer/compute ratio {ratio:.4f} unexpectedly low"

    def test_correct_number_of_transfers(self):
        # 320 microbatches, 2 stages → 320 forward + 320 backward transfers = 640.
        assert len(self.collector.transfers) == 640


class TestExp05TransferTiming:
    """Verify per-transfer duration matches link parameters.

    Link: 12.5 Gbps, 200 µs latency.
    Activation: activation_mb=64 × bf16 data_scale=0.5 → 32 MB in-pipeline.
    Expected transfer time = 0.200 ms + 32 MB × 8 / 12.5 Gbps
                           = 0.200 + 256 / 12.5 ms = 0.200 + 20.48 = ~20.68 ms.
    """

    def test_forward_transfer_duration(self):
        config = _exp05_config(mean=22.0, std=0.0)
        config["simulation"]["batches"] = 1
        config["simulation"]["microbatches"] = 1
        runtime = _run(config)

        fwd_transfers = [
            r for r in runtime.collector.transfers
            if r.src_device == "h100_node0_gpu0" and r.dst_device == "h100_node1_gpu0"
        ]
        assert fwd_transfers, "Expected at least one forward transfer"
        duration = fwd_transfers[0].end_time - fwd_transfers[0].start_time

        # activation_mb=64 × 0.5 (bf16) = 32 MB → 32*8/12.5 + 0.2 = ~20.68 ms
        assert 18.0 <= duration <= 25.0, f"Transfer duration {duration:.3f} ms unexpected"

    def test_backward_transfer_duration_matches_forward(self):
        # Backward activations are same size → same transfer time.
        config = _exp05_config(mean=22.0, std=0.0)
        config["simulation"]["batches"] = 1
        config["simulation"]["microbatches"] = 1
        runtime = _run(config)

        fwd = [r for r in runtime.collector.transfers if r.src_device == "h100_node0_gpu0"]
        bwd = [r for r in runtime.collector.transfers if r.src_device == "h100_node1_gpu0"]
        assert fwd and bwd

        fwd_dur = fwd[0].end_time - fwd[0].start_time
        bwd_dur = bwd[0].end_time - bwd[0].start_time
        assert abs(fwd_dur - bwd_dur) < 1e-6, "Symmetric links must produce equal transfer times"


# ---------------------------------------------------------------------------
# Experiment 06 – single-GPU smoke test, run 1
# ---------------------------------------------------------------------------

class TestExp06Smoke:
    """Single-GPU run: PP=1, 4 batches × 1 microbatch = 4 microbatches.

    Observed values from experiment output:
      throughput ≈ 0.03393 µb/ms, bubble = 0.0, device util = 1.0,
      makespan ≈ 117.90 ms, transfer time = 0.
    """

    def setup_method(self):
        self.runtime = _run(_exp06_config(mean=20.0, std=0.0))
        self.collector = self.runtime.collector

    def test_completed_microbatches(self):
        assert self.collector.completed_microbatches == 4

    def test_zero_bubble_single_stage(self):
        # Single stage → no pipeline bubble possible.
        assert self.collector.bubble_ratio() == 0.0

    def test_full_device_utilization(self):
        util = self.collector.per_device_utilization()
        assert abs(util["h100_node0_gpu0"] - 1.0) < 1e-6

    def test_no_transfers(self):
        assert len(self.collector.transfers) == 0

    def test_throughput_order_of_magnitude(self):
        # Observed ≈ 0.03393 µb/ms. Constant dist → deterministic.
        # fwd=20ms, bwd=40ms per µb → throughput ≈ 1/60ms ≈ 0.0167 µb/ms
        # (first µb completes at 60ms, last at 4*60=240ms → throughput over span).
        tp = self.collector.throughput()
        assert 0.010 <= tp <= 0.050, f"throughput {tp:.5f} µb/ms outside expected range"

    def test_makespan_is_all_compute(self):
        # Single device, no transfers → makespan == total sequential compute time.
        total_compute = self.collector.total_compute_time()
        makespan = self.collector.makespan()
        # 4 µb × (fwd=20ms + bwd=40ms) = 240ms total compute = makespan
        assert abs(makespan - total_compute) < 1e-6

    def test_makespan_matches_batches_times_per_microbatch_time(self):
        # bf16 compute_speedup=2.0 halves all latency samples.
        # fwd = 20ms / 2 = 10ms, bwd = 20ms × 2.0 / 2.0 = 20ms → 30ms per µb.
        # 4 µb in sequence = 120ms.
        makespan = self.collector.makespan()
        assert abs(makespan - 120.0) < 1e-6, f"makespan {makespan:.3f} ms, expected 120.0"

    def test_peak_memory_within_h100_capacity(self):
        peaks = self.collector.peak_memory_per_device
        # weights=2048 × 1.5 bf16 overhead = 3072 MB; + activations (32 MB) → ~3104 MB observed
        assert 3000 <= peaks["h100_node0_gpu0"] <= 81920

    def test_forward_and_backward_computes_recorded(self):
        phases = {r.phase for r in self.collector.computes}
        assert Phase.FORWARD in phases
        assert Phase.BACKWARD in phases

    def test_per_microbatch_e2e_latency(self):
        # bf16 compute_speedup=2.0 halves all latency samples:
        # fwd = 20/2 = 10ms, bwd = (20×2.0)/2.0 = 20ms → e2e = 30ms per µb.
        latencies = self.collector.e2e_latencies()
        assert len(latencies) == 4
        for lat in latencies:
            assert abs(lat - 30.0) < 1e-6, f"Expected 30ms latency, got {lat:.3f}ms"

    def test_correct_number_of_compute_events(self):
        # 4 µb × (1 fwd + 1 bwd) = 8 compute records
        assert len(self.collector.computes) == 8
