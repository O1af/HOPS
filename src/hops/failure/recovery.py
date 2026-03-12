"""Recovery strategies: re-scheduling, backpressure adjustments."""

from abc import ABC, abstractmethod

from hops.core.event_engine import EventEngine
from hops.core.types import Event


class RecoveryStrategy(ABC):
    """Base class for failure recovery strategies."""

    @abstractmethod
    def recover(self, event: Event, engine: EventEngine) -> None:
        """Handle recovery after a failure event."""


class RestartRecovery(RecoveryStrategy):
    """Re-execute the affected micro-batch after a delay."""

    def __init__(self, delay: float = 5.0):
        self.delay = delay

    def recover(self, event: Event, engine: EventEngine) -> None:
        # The failure engine handles scheduling RECOVERY events directly.
        # This class is available for custom recovery logic extensions.
        pass


class RerouteRecovery(RecoveryStrategy):
    """Reassign stage to a backup device."""

    def __init__(self, backup_mapping: dict[str, str] | None = None):
        self.backup_mapping = backup_mapping or {}

    def recover(self, event: Event, engine: EventEngine) -> None:
        # Extension point: remap failed device to backup
        pass
