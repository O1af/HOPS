"""GPU and CPU device abstractions."""

from dataclasses import dataclass


@dataclass
class Device:
    id: str
    kind: str  # "gpu" or "cpu"
    flops: float  # peak TFLOPS
    memory_mb: float
    memory_bandwidth_gbps: float
    numa_node: int = 0
    busy_until: float = 0.0

    @classmethod
    def from_yaml(cls, config: dict) -> "Device":
        return cls(
            id=config["id"],
            kind=config["kind"],
            flops=config["flops"],
            memory_mb=config["memory_mb"],
            memory_bandwidth_gbps=config["memory_bandwidth_gbps"],
            numa_node=config.get("numa_node", 0),
        )
