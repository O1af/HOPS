"""Discrete event simulation engine."""

import heapq
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from hops.core.types import Event, EventKind


class EventEngine:
    """Priority-queue event loop with handler registration."""

    def __init__(self):
        self._queue: list[tuple[float, int, Event]] = []
        self._now: float = 0.0
        self._handlers: dict[EventKind, list[Callable]] = defaultdict(list)
        self._sequence: int = 0

    @property
    def now(self) -> float:
        return self._now

    def schedule(self, event: Event) -> None:
        heapq.heappush(self._queue, (event.time, self._sequence, event))
        self._sequence += 1

    def on(self, kind: EventKind, handler: Callable[[Event, "EventEngine"], None]) -> None:
        """Register a handler for an event kind."""
        self._handlers[kind].append(handler)

    def run(self, until: float = float("inf"),
            stop_condition: Callable[[], Any] | None = None) -> None:
        """Process events until the queue is empty or time exceeds `until`."""
        while self._queue:
            if stop_condition is not None and stop_condition():
                break
            if self._queue[0][0] > until:
                break
            _, _, event = heapq.heappop(self._queue)
            self._now = event.time
            for handler in self._handlers[event.kind]:
                handler(event, self)

    def clear(self) -> None:
        """Reset engine state for a new batch."""
        self._queue.clear()

    def pending(self) -> int:
        return len(self._queue)
