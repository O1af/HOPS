"""GPU and CPU device abstractions."""

from dataclasses import dataclass


@dataclass
class Device:
    id: str
    kind: str  # "gpu" or "cpu"
    memory_mb: float
    busy_until: float = 0.0

    @classmethod
    def from_yaml(cls, config: dict) -> "Device":
        return cls(
            id=config["id"],
            kind=config["kind"],
            memory_mb=config["memory_mb"],
        )
