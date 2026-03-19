"""GPU and CPU device abstractions."""

from dataclasses import dataclass


def _infer_node_id(device_id: str) -> str:
    if "_" in device_id:
        return device_id.split("_", 1)[0]
    return "node0"


def _infer_numa_node(config: dict) -> int:
    if "numa_node" in config:
        return int(config["numa_node"])
    socket_id = str(config.get("socket_id", "socket0"))
    digits = "".join(ch for ch in socket_id if ch.isdigit())
    return int(digits) if digits else 0


@dataclass
class Device:
    id: str
    kind: str  # "gpu" or "cpu"
    memory_mb: float
    flops: float | None = None
    memory_bandwidth_gbps: float | None = None
    node_id: str = "node0"
    socket_id: str = "socket0"
    numa_node: int = 0
    busy_until: float = 0.0
    memory_used_mb: float = 0.0
    peak_memory_mb: float = 0.0

    def allocate(self, size_mb: float) -> None:
        self.memory_used_mb += size_mb
        if self.memory_used_mb > self.peak_memory_mb:
            self.peak_memory_mb = self.memory_used_mb

    def free(self, size_mb: float) -> None:
        self.memory_used_mb = max(0.0, self.memory_used_mb - size_mb)

    @classmethod
    def from_yaml(cls, config: dict) -> "Device":
        numa_node = _infer_numa_node(config)
        node_id = str(config.get("node_id", _infer_node_id(config["id"])))
        socket_id = str(config.get("socket_id", f"socket{numa_node}"))
        return cls(
            id=config["id"],
            kind=config["kind"],
            memory_mb=config["memory_mb"],
            flops=config.get("flops"),
            memory_bandwidth_gbps=config.get("memory_bandwidth_gbps"),
            node_id=node_id,
            socket_id=socket_id,
            numa_node=numa_node,
        )
