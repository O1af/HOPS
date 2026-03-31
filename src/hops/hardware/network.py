"""Inter-node and intra-node communication links."""

from dataclasses import dataclass

from hops.latency.distributions import Distribution


@dataclass
class Link:
    src: str
    dst: str
    bandwidth_gbps: float
    base_latency_us: float
    jitter: Distribution
    active_transfers: int = 0
