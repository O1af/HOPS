"""Built-in hardware and interconnect preset catalogs."""

from __future__ import annotations

from dataclasses import dataclass

from hops.hardware.topology import LocalityPenalty


@dataclass(frozen=True)
class DevicePreset:
    kind: str
    flops_tflops: float
    memory_mb: float
    memory_bandwidth_gbps: float
    # Per-invocation overhead (kernel launch + Python/CUDA glue). Dominates
    # sub-ms kernels on modern GPUs where the FLOP-budget estimate underfits.
    launch_overhead_ms: float = 1.5


@dataclass(frozen=True)
class InterconnectPreset:
    bandwidth_gbps: float
    latency_us: float
    jitter: dict
    penalty: LocalityPenalty = LocalityPenalty()


DEVICE_PRESETS: dict[str, DevicePreset] = {
    "h100": DevicePreset(
        kind="gpu",
        flops_tflops=989.0,
        memory_mb=81920.0,
        memory_bandwidth_gbps=2000.0,
    ),
    "a100": DevicePreset(
        kind="gpu",
        flops_tflops=312.0,
        memory_mb=81920.0,
        memory_bandwidth_gbps=2039.0,
    ),
    "a10g": DevicePreset(
        kind="gpu",
        flops_tflops=125.0,
        memory_mb=24576.0,
        memory_bandwidth_gbps=600.0,
    ),
    "l4": DevicePreset(
        kind="gpu",
        flops_tflops=121.0,
        memory_mb=24576.0,
        memory_bandwidth_gbps=300.0,
    ),
    "l40s": DevicePreset(
        kind="gpu",
        flops_tflops=362.0,
        memory_mb=49152.0,
        memory_bandwidth_gbps=864.0,
    ),
    "cpu-standard": DevicePreset(
        kind="cpu",
        flops_tflops=8.0,
        memory_mb=262144.0,
        memory_bandwidth_gbps=200.0,
    ),
}


INTERCONNECT_PRESETS: dict[str, InterconnectPreset] = {
    "nvlink": InterconnectPreset(
        bandwidth_gbps=4800.0,
        latency_us=1.0,
        jitter={"type": "normal", "mean": 0.0, "std": 0.05},
        penalty=LocalityPenalty(
            compute_scale=1.02,
            memory_bandwidth_scale=0.95,
            memory_latency_us=1.0,
            transfer_scale=1.0,
        ),
    ),
    "pcie": InterconnectPreset(
        bandwidth_gbps=256.0,
        latency_us=2.5,
        jitter={"type": "normal", "mean": 0.0, "std": 0.1},
        penalty=LocalityPenalty(
            compute_scale=1.05,
            memory_bandwidth_scale=0.85,
            memory_latency_us=3.0,
            transfer_scale=1.05,
        ),
    ),
    "infiniband": InterconnectPreset(
        bandwidth_gbps=200.0,
        latency_us=5.0,
        jitter={"type": "normal", "mean": 0.0, "std": 0.3},
        penalty=LocalityPenalty(
            compute_scale=1.10,
            memory_bandwidth_scale=0.70,
            memory_latency_us=8.0,
            transfer_scale=1.20,
        ),
    ),
    "ethernet": InterconnectPreset(
        bandwidth_gbps=100.0,
        latency_us=15.0,
        jitter={"type": "normal", "mean": 0.0, "std": 1.0},
        penalty=LocalityPenalty(
            compute_scale=1.15,
            memory_bandwidth_scale=0.60,
            memory_latency_us=12.0,
            transfer_scale=1.40,
        ),
    ),
}


class PresetRegistry:
    """Resolve named presets into concrete hardware characteristics."""

    def __init__(self,
                 device_presets: dict[str, DevicePreset] | None = None,
                 interconnect_presets: dict[str, InterconnectPreset] | None = None):
        self._device_presets = device_presets if device_presets is not None else dict(DEVICE_PRESETS)
        self._interconnect_presets = (
            interconnect_presets if interconnect_presets is not None else dict(INTERCONNECT_PRESETS)
        )

    def device(self, name: str) -> DevicePreset:
        if name not in self._device_presets:
            raise ValueError(
                f"Unknown hardware preset {name!r}. "
                f"Available presets: {sorted(self._device_presets)}"
            )
        return self._device_presets[name]

    def interconnect(self, name: str) -> InterconnectPreset:
        if name not in self._interconnect_presets:
            raise ValueError(
                f"Unknown interconnect preset {name!r}. "
                f"Available presets: {sorted(self._interconnect_presets)}"
            )
        return self._interconnect_presets[name]
