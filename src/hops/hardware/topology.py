"""Node and device graph with NUMA modeling."""

from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.latency.distributions import Constant

_ZERO_JITTER = Constant(0.0)


class Topology:
    """Directed graph of devices and communication links."""

    def __init__(self, devices: list[Device], links: list[Link]):
        self._devices = {d.id: d for d in devices}
        self._links = {(lnk.src, lnk.dst): lnk for lnk in links}
        # Pre-cache same-device links to avoid allocation on hot path
        self._self_links: dict[str, Link] = {
            d.id: Link(d.id, d.id, bandwidth_gbps=float("inf"),
                       base_latency_us=0.0, jitter=_ZERO_JITTER)
            for d in devices
        }

    def device(self, device_id: str) -> Device:
        return self._devices[device_id]

    def link(self, src: str, dst: str) -> Link:
        if (src, dst) in self._links:
            return self._links[(src, dst)]
        if src == dst:
            return self._self_links[src]
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
        return cls(devices, links)
