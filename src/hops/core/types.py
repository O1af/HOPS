"""Central type definitions for the HOPS simulator."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventKind(Enum):
    COMPUTE_START = auto()
    COMPUTE_END = auto()
    TRANSFER_START = auto()
    TRANSFER_END = auto()
    FAILURE = auto()
    RECOVERY = auto()


class Phase(Enum):
    FORWARD = auto()
    BACKWARD = auto()


@dataclass(order=True)
class Event:
    time: float
    kind: EventKind = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class MicroBatch:
    id: int
    global_batch_id: int = 0


@dataclass
class StageTask:
    microbatch: MicroBatch
    stage_id: int
    phase: Phase
