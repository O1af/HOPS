"""Import raw Megatron trace files into HOPS metrics structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from hops.core.types import Phase
from hops.metrics.collector import MetricsCollector


_SUPPORTED_EVENT_TYPES = {"compute", "transfer"}


@dataclass(frozen=True)
class RawMegatronEvent:
    rank: int
    stage: int
    iteration: int
    microbatch: int | None
    event_type: str
    phase: Phase
    start_wall_ns: int
    end_wall_ns: int
    device_id: str
    src_device: str | None = None
    dst_device: str | None = None
    hostname: str | None = None

    @property
    def time_key(self) -> tuple[int, int, int, int]:
        return (
            self.start_wall_ns,
            self.end_wall_ns,
            self.stage,
            self.rank,
        )


def _require(mapping: dict, field: str):
    if field not in mapping:
        raise ValueError(f"Missing required field {field!r} in Megatron trace event")
    return mapping[field]


def _parse_phase(raw: str) -> Phase:
    try:
        return Phase[raw]
    except KeyError:
        raise ValueError(
            f"Unsupported phase {raw!r}; expected one of {sorted(Phase.__members__)}"
        )


def _parse_event(data: dict) -> RawMegatronEvent:
    event_type = _require(data, "event_type")
    if event_type not in _SUPPORTED_EVENT_TYPES:
        raise ValueError(
            f"Unsupported event_type {event_type!r}; expected one of {sorted(_SUPPORTED_EVENT_TYPES)}"
        )

    start_wall_ns = int(_require(data, "start_wall_ns"))
    end_wall_ns = int(_require(data, "end_wall_ns"))
    if end_wall_ns < start_wall_ns:
        raise ValueError("Megatron trace event end_wall_ns must be >= start_wall_ns")

    microbatch = data.get("microbatch")
    if microbatch is not None:
        microbatch = int(microbatch)

    src_device = data.get("src_device")
    dst_device = data.get("dst_device")
    if event_type == "transfer" and (not src_device or not dst_device):
        raise ValueError("Transfer events must define src_device and dst_device")

    return RawMegatronEvent(
        rank=int(_require(data, "rank")),
        stage=int(_require(data, "stage")),
        iteration=int(_require(data, "iteration")),
        microbatch=microbatch,
        event_type=event_type,
        phase=_parse_phase(str(_require(data, "phase"))),
        start_wall_ns=start_wall_ns,
        end_wall_ns=end_wall_ns,
        device_id=str(_require(data, "device_id")),
        src_device=str(src_device) if src_device else None,
        dst_device=str(dst_device) if dst_device else None,
        hostname=str(data["hostname"]) if "hostname" in data else None,
    )


def load_raw_megatron_events(trace_dir: str | Path) -> list[RawMegatronEvent]:
    path = Path(trace_dir)
    if not path.exists():
        raise FileNotFoundError(f"Megatron trace directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Megatron trace path is not a directory: {path}")

    events: list[RawMegatronEvent] = []
    for trace_file in sorted(path.glob("*.jsonl")):
        with trace_file.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {trace_file}:{line_no}: {exc}"
                    ) from exc
                events.append(_parse_event(payload))

    if not events:
        raise ValueError(f"No Megatron trace events found in {path}")
    events.sort(key=lambda event: event.time_key)
    return events


def _global_microbatch_ids(events: list[RawMegatronEvent]) -> dict[tuple[int, int], int]:
    pairs = sorted({
        (event.iteration, event.microbatch)
        for event in events
        if event.microbatch is not None
    })
    return {pair: idx for idx, pair in enumerate(pairs)}


def import_megatron_trace_dir(trace_dir: str | Path) -> MetricsCollector:
    events = load_raw_megatron_events(trace_dir)
    origin_ns = events[0].start_wall_ns  # list is sorted by time_key whose first element is start_wall_ns
    global_mb_ids = _global_microbatch_ids(events)
    collector = MetricsCollector()

    for event in events:
        start_ms = (event.start_wall_ns - origin_ns) / 1_000_000.0
        end_ms = (event.end_wall_ns - origin_ns) / 1_000_000.0
        global_mb_id = None
        if event.microbatch is not None:
            global_mb_id = global_mb_ids[(event.iteration, event.microbatch)]

        if event.event_type == "compute":
            collector.record_compute(
                stage_id=event.stage,
                microbatch_id=global_mb_id,
                phase=event.phase,
                device_id=event.device_id,
                start_time=start_ms,
                end_time=end_ms,
            )
            if event.stage == 0 and event.phase == Phase.BACKWARD and global_mb_id is not None:
                collector.record_microbatch_completion(global_mb_id, end_ms)
        else:
            collector.record_transfer(
                microbatch_id=global_mb_id,
                phase=event.phase,
                src=event.src_device or "",
                dst=event.dst_device or "",
                start=start_ms,
                end=end_ms,
            )

    return collector
