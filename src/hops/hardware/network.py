"""Inter-node and intra-node communication links."""

from dataclasses import dataclass

import numpy as np

from hops.latency.distributions import Distribution


@dataclass
class Link:
    src: str
    dst: str
    bandwidth_gbps: float
    base_latency_us: float
    jitter: Distribution

    def sample_transfer_time(self, size_mb: float, rng: np.random.Generator) -> float:
        """Return total transfer time in simulation time units (ms)."""
        # Convert base latency from microseconds to ms
        base_ms = self.base_latency_us / 1000.0
        # size_mb -> Gbit: size_mb * 8 / 1000; time_s = Gbit / Gbps; time_ms = time_s * 1000
        transfer_ms = (size_mb * 8.0) / self.bandwidth_gbps
        return base_ms + transfer_ms + self.jitter.sample(rng)

    @classmethod
    def from_yaml(cls, config: dict) -> "Link":
        return cls(
            src=config["src"],
            dst=config["dst"],
            bandwidth_gbps=config["bandwidth_gbps"],
            base_latency_us=config["base_latency_us"],
            jitter=Distribution.from_yaml(config["jitter"]),
        )
