"""Chaos Monkey-style failure injection engine."""

from __future__ import annotations

import numpy as np

from hops.config import FailureConfig
from hops.core.event_engine import EventEngine
from hops.core.types import Event, EventKind
from hops.hardware.topology import Topology
from hops.metrics.collector import MetricsCollector


class FailureEngine:
    """Periodically checks for and injects device/link failures."""

    def __init__(self, engine: EventEngine, topology: Topology,
                 collector: MetricsCollector, config: FailureConfig,
                 rng: np.random.Generator):
        self.engine = engine
        self.topology = topology
        self.collector = collector
        self.rng = rng
        self.check_interval = config.check_interval_ms
        self.device_fail_prob = config.device_failure_probability
        self.link_fail_prob = config.link_failure_probability
        self.recovery_time = config.recovery_time_ms
        self._failed_devices: set[str] = set()
        self._failed_links: set[tuple[str, str]] = set()
        self._device_recovery_times: dict[str, float] = {}
        self._link_recovery_times: dict[tuple[str, str], float] = {}

        engine.on(EventKind.FAILURE, self._on_failure)
        engine.on(EventKind.RECOVERY, self._on_recovery)

        # Schedule first check
        self._schedule_next_check()

    def _schedule_next_check(self) -> None:
        self.engine.schedule(Event(
            time=self.engine.now + self.check_interval,
            kind=EventKind.FAILURE,
            payload={"type": "check"},
        ))

    def _on_failure(self, event: Event, engine: EventEngine) -> None:
        if event.payload.get("type") == "check":
            self._do_failure_check(engine)
            self._schedule_next_check()
            return

        recovery_at = engine.now + self.recovery_time
        if event.payload["target_type"] == "device":
            device_id = event.payload["device_id"]
            self._failed_devices.add(device_id)
            self._device_recovery_times[device_id] = recovery_at
            self.collector.record_failure(device_id, engine.now, self.recovery_time)
            payload = {"target_type": "device", "device_id": device_id}
        else:
            link_id = (event.payload["src"], event.payload["dst"])
            self._failed_links.add(link_id)
            self._link_recovery_times[link_id] = recovery_at
            self.collector.record_failure(f"{link_id[0]}->{link_id[1]}", engine.now, self.recovery_time)
            payload = {"target_type": "link", "src": link_id[0], "dst": link_id[1]}

        # Schedule recovery
        engine.schedule(Event(
            time=recovery_at,
            kind=EventKind.RECOVERY,
            payload=payload,
        ))

    def _do_failure_check(self, engine: EventEngine) -> None:
        for device_id in self.topology.devices:
            if device_id in self._failed_devices:
                continue
            if self.rng.random() < self.device_fail_prob:
                engine.schedule(Event(
                    time=engine.now,
                    kind=EventKind.FAILURE,
                    payload={"target_type": "device", "device_id": device_id},
                ))
        for src, dst in self.topology.links:
            if (src, dst) in self._failed_links:
                continue
            if self.rng.random() < self.link_fail_prob:
                engine.schedule(Event(
                    time=engine.now,
                    kind=EventKind.FAILURE,
                    payload={"target_type": "link", "src": src, "dst": dst},
                ))

    def _on_recovery(self, event: Event, engine: EventEngine) -> None:
        if event.payload["target_type"] == "device":
            device_id = event.payload["device_id"]
            self._failed_devices.discard(device_id)
            self._device_recovery_times.pop(device_id, None)
        else:
            link_id = (event.payload["src"], event.payload["dst"])
            self._failed_links.discard(link_id)
            self._link_recovery_times.pop(link_id, None)

    def is_failed(self, device_id: str) -> bool:
        return device_id in self._failed_devices

    def is_device_failed(self, device_id: str) -> bool:
        return device_id in self._failed_devices

    def is_link_failed(self, src: str, dst: str) -> bool:
        return (src, dst) in self._failed_links

    def next_device_recovery_time(self, device_id: str) -> float | None:
        return self._device_recovery_times.get(device_id)

    def next_link_recovery_time(self, src: str, dst: str) -> float | None:
        return self._link_recovery_times.get((src, dst))
