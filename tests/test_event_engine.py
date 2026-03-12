"""Tests for the discrete event simulation engine."""

from hops.core.event_engine import EventEngine
from hops.core.types import Event, EventKind


def test_events_processed_in_time_order():
    engine = EventEngine()
    order = []
    engine.on(EventKind.COMPUTE_START, lambda e, eng: order.append(e.time))

    engine.schedule(Event(time=3.0, kind=EventKind.COMPUTE_START))
    engine.schedule(Event(time=1.0, kind=EventKind.COMPUTE_START))
    engine.schedule(Event(time=2.0, kind=EventKind.COMPUTE_START))
    engine.run()

    assert order == [1.0, 2.0, 3.0]


def test_handlers_can_schedule_new_events():
    engine = EventEngine()
    results = []

    def handler(event, eng):
        results.append(event.time)
        if event.time < 5.0:
            eng.schedule(Event(time=event.time + 2.0, kind=EventKind.COMPUTE_START))

    engine.on(EventKind.COMPUTE_START, handler)
    engine.schedule(Event(time=1.0, kind=EventKind.COMPUTE_START))
    engine.run()

    assert results == [1.0, 3.0, 5.0]


def test_run_until_stops_at_time():
    engine = EventEngine()
    processed = []
    engine.on(EventKind.COMPUTE_END, lambda e, eng: processed.append(e.time))

    for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
        engine.schedule(Event(time=t, kind=EventKind.COMPUTE_END))

    engine.run(until=3.5)
    assert processed == [1.0, 2.0, 3.0]
    assert engine.pending() == 2


def test_now_tracks_current_time():
    engine = EventEngine()
    times = []
    engine.on(EventKind.COMPUTE_START, lambda e, eng: times.append(eng.now))

    engine.schedule(Event(time=5.0, kind=EventKind.COMPUTE_START))
    engine.schedule(Event(time=10.0, kind=EventKind.COMPUTE_START))
    engine.run()

    assert times == [5.0, 10.0]


def test_multiple_handlers_per_event_kind():
    engine = EventEngine()
    results_a = []
    results_b = []
    engine.on(EventKind.COMPUTE_START, lambda e, eng: results_a.append(e.time))
    engine.on(EventKind.COMPUTE_START, lambda e, eng: results_b.append(e.time))

    engine.schedule(Event(time=1.0, kind=EventKind.COMPUTE_START))
    engine.run()

    assert results_a == [1.0]
    assert results_b == [1.0]
