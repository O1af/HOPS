"""Discrete event simulation engine."""

import heapq
from collections import defaultdict
from collections.abc import Callable

from hops.core.types import Event, EventKind


class EventEngine:
    """Priority-queue event loop with handler registration."""

    def __init__(self):
        self._queue: list[Event] = []
        self._now: float = 0.0
        self._handlers: dict[EventKind, list[Callable]] = defaultdict(list)

    @property
    def now(self) -> float:
        return self._now

    def schedule(self, event: Event) -> None:
        heapq.heappush(self._queue, event)

    def on(self, kind: EventKind, handler: Callable[[Event, "EventEngine"], None]) -> None:
        """Register a handler for an event kind."""
        self._handlers[kind].append(handler)

    def run(self, until: float = float("inf")) -> None:
        """Process events until the queue is empty or time exceeds `until`."""
        while self._queue:
            if self._queue[0].time > until:
                break
            event = heapq.heappop(self._queue)
            self._now = event.time
            for handler in self._handlers[event.kind]:
                handler(event, self)

    def clear(self) -> None:
        """Reset engine state for a new batch."""
        self._queue.clear()

    def pending(self) -> int:
        return len(self._queue)
