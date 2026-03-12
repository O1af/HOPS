"""Chaos Monkey-style failure injection engine."""

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.types import Event, EventKind
from hops.hardware.topology import Topology
from hops.metrics.collector import MetricsCollector


class FailureEngine:
    """Periodically checks for and injects device/link failures."""

    def __init__(self, engine: EventEngine, topology: Topology,
                 collector: MetricsCollector, config: dict):
        self.engine = engine
        self.topology = topology
        self.collector = collector
        self.check_interval = config.get("check_interval", 10.0)
        self.device_fail_prob = config.get("device_fail_prob", 0.001)
        self.link_fail_prob = config.get("link_fail_prob", 0.0005)
        self.recovery_time = config.get("recovery_time", 5.0)
        self._failed_devices: set[str] = set()

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

        # Actual device failure
        device_id = event.payload["device_id"]
        self._failed_devices.add(device_id)
        self.collector.record_failure(device_id, engine.now, self.recovery_time)

        # Schedule recovery
        engine.schedule(Event(
            time=engine.now + self.recovery_time,
            kind=EventKind.RECOVERY,
            payload={"device_id": device_id},
        ))

    def _do_failure_check(self, engine: EventEngine) -> None:
        for device_id in self.topology.devices:
            if device_id in self._failed_devices:
                continue
            if np.random.random() < self.device_fail_prob:
                engine.schedule(Event(
                    time=engine.now,
                    kind=EventKind.FAILURE,
                    payload={"device_id": device_id},
                ))

    def _on_recovery(self, event: Event, engine: EventEngine) -> None:
        device_id = event.payload["device_id"]
        self._failed_devices.discard(device_id)

    def is_failed(self, device_id: str) -> bool:
        return device_id in self._failed_devices
