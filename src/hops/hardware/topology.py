"""Node and device graph with NUMA modeling."""

from dataclasses import dataclass
from enum import Enum

from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.latency.distributions import Constant
from hops.latency.distributions import Distribution

_ZERO_JITTER = Constant(0.0)


class Locality(Enum):
    SAME_DEVICE = "same_device"
    SAME_SOCKET = "same_socket"
    SAME_NODE = "same_node"
    CROSS_NODE = "cross_node"


@dataclass(frozen=True)
class LinkProfile:
    bandwidth_gbps: float
    base_latency_us: float
    jitter: Distribution

    @classmethod
    def from_yaml(cls, config: dict) -> "LinkProfile":
        return cls(
            bandwidth_gbps=config["bandwidth_gbps"],
            base_latency_us=config["base_latency_us"],
            jitter=Distribution.from_yaml(config["jitter"]),
        )


@dataclass(frozen=True)
class LocalityPenalty:
    compute_scale: float = 1.0
    memory_bandwidth_scale: float = 1.0
    memory_latency_us: float = 0.0
    transfer_scale: float = 1.0

    @classmethod
    def from_yaml(cls, config: dict) -> "LocalityPenalty":
        return cls(
            compute_scale=config.get("compute_scale", 1.0),
            memory_bandwidth_scale=config.get("memory_bandwidth_scale", 1.0),
            memory_latency_us=config.get("memory_latency_us", 0.0),
            transfer_scale=config.get("transfer_scale", 1.0),
        )


class Topology:
    """Directed graph of devices and communication links."""

    def __init__(self, devices: list[Device], links: list[Link],
                 link_profiles: dict[Locality, LinkProfile] | None = None,
                 locality_penalties: dict[Locality, LocalityPenalty] | None = None):
        self._devices = {d.id: d for d in devices}
        self._links = {(lnk.src, lnk.dst): lnk for lnk in links}
        self._link_profiles = link_profiles or {}
        self._locality_penalties = locality_penalties or {}
        # Pre-cache same-device links to avoid allocation on hot path
        self._self_links: dict[str, Link] = {
            d.id: Link(d.id, d.id, bandwidth_gbps=float("inf"),
                       base_latency_us=0.0, jitter=_ZERO_JITTER)
            for d in devices
        }
        self._derived_links: dict[tuple[str, str], Link] = {}

    def locality(self, src: str, dst: str) -> Locality:
        if src == dst:
            return Locality.SAME_DEVICE
        src_device = self.device(src)
        dst_device = self.device(dst)
        if src_device.node_id == dst_device.node_id:
            if src_device.socket_id == dst_device.socket_id:
                return Locality.SAME_SOCKET
            return Locality.SAME_NODE
        return Locality.CROSS_NODE

    def device(self, device_id: str) -> Device:
        return self._devices[device_id]

    def locality_from_placement(self, *, device_id: str,
                                node_id: str | None = None,
                                socket_id: str | None = None) -> Locality:
        device = self.device(device_id)
        target_node = node_id or device.node_id
        target_socket = socket_id or device.socket_id
        if target_node == device.node_id:
            if target_socket == device.socket_id:
                return Locality.SAME_SOCKET
            return Locality.SAME_NODE
        return Locality.CROSS_NODE

    def locality_penalty(self, locality: Locality) -> LocalityPenalty:
        return self._locality_penalties.get(locality, LocalityPenalty())

    def stage_locality_penalty(self, *, device_id: str, memory_placement=None) -> LocalityPenalty:
        if memory_placement is None or getattr(memory_placement, "kind", "local") == "local":
            return self.locality_penalty(Locality.SAME_SOCKET)
        if memory_placement.kind == "socket":
            locality = self.locality_from_placement(
                device_id=device_id,
                node_id=memory_placement.node,
                socket_id=str(memory_placement.socket),
            )
            return self.locality_penalty(locality)
        if memory_placement.kind == "device":
            locality = self.locality(device_id, memory_placement.device)
            return self.locality_penalty(locality)
        return self.locality_penalty(Locality.SAME_SOCKET)

    def transfer_penalty(self, src: str, dst: str) -> LocalityPenalty:
        return self.locality_penalty(self.locality(src, dst))

    def link(self, src: str, dst: str) -> Link:
        if (src, dst) in self._links:
            return self._links[(src, dst)]
        if src == dst:
            return self._self_links[src]
        if (src, dst) in self._derived_links:
            return self._derived_links[(src, dst)]
        locality = self.locality(src, dst)
        profile = self._link_profiles.get(locality)
        if profile is not None:
            link = Link(
                src=src,
                dst=dst,
                bandwidth_gbps=profile.bandwidth_gbps,
                base_latency_us=profile.base_latency_us,
                jitter=profile.jitter,
            )
            self._derived_links[(src, dst)] = link
            return link
        raise KeyError(f"No link from {src} to {dst}")

    @property
    def devices(self) -> dict[str, Device]:
        return self._devices

    @property
    def links(self) -> dict[tuple[str, str], Link]:
        return self._links

    @classmethod
    def from_yaml(cls, config: dict) -> "Topology":
        devices = [Device.from_yaml(d) for d in config["devices"]]
        links = [Link.from_yaml(lnk) for lnk in config.get("links", [])]
        fabric = config.get("fabric", {})
        locality_penalty_cfg = config.get("locality_penalties", config.get("numa_penalties", {}))
        profiles: dict[Locality, LinkProfile] = {}
        penalties: dict[Locality, LocalityPenalty] = {}
        aliases = {
            Locality.SAME_SOCKET: ("same_socket", "intra_socket"),
            Locality.SAME_NODE: ("same_node", "intra_node"),
            Locality.CROSS_NODE: ("cross_node", "inter_node"),
        }
        for locality, keys in aliases.items():
            for key in keys:
                if key in fabric:
                    profiles[locality] = LinkProfile.from_yaml(fabric[key])
                    break
            for key in keys:
                if key in locality_penalty_cfg:
                    penalties[locality] = LocalityPenalty.from_yaml(locality_penalty_cfg[key])
                    break
        return cls(devices, links, link_profiles=profiles, locality_penalties=penalties)
