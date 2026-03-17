"""GPU and CPU device abstractions."""

from dataclasses import dataclass


@dataclass
class Device:
    id: str
    kind: str  # "gpu" or "cpu"
    memory_mb: float
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
        return cls(
            id=config["id"],
            kind=config["kind"],
            memory_mb=config["memory_mb"],
        )
