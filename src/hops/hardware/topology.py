"""Node and device graph with NUMA modeling."""

from hops.hardware.device import Device
from hops.hardware.network import Link
from hops.latency.distributions import Constant


class Topology:
    """Directed graph of devices and communication links."""

    def __init__(self, devices: list[Device], links: list[Link]):
        self._devices = {d.id: d for d in devices}
        self._links = {(l.src, l.dst): l for l in links}

    def device(self, device_id: str) -> Device:
        return self._devices[device_id]

    def link(self, src: str, dst: str) -> Link:
        if (src, dst) in self._links:
            return self._links[(src, dst)]
        # If same device, zero-cost transfer
        if src == dst:
            return Link(src, dst, bandwidth_gbps=float("inf"),
                        base_latency_us=0.0, jitter=Constant(0.0))
        raise KeyError(f"No link from {src} to {dst}")

    @property
    def devices(self) -> dict[str, Device]:
        return self._devices

    @classmethod
    def from_yaml(cls, config: dict) -> "Topology":
        devices = [Device.from_yaml(d) for d in config["devices"]]
        links = [Link.from_yaml(l) for l in config.get("links", [])]
        return cls(devices, links)
