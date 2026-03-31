"""GPU and CPU device abstractions."""

from dataclasses import dataclass


def numa_from_socket(socket_id: str) -> int:
    """Extract a NUMA node index from a socket identifier string."""
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
