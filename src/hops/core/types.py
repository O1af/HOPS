"""Central type definitions for the HOPS simulator."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventKind(Enum):
    COMPUTE_START = auto()
    COMPUTE_END = auto()
    TRANSFER_START = auto()
    TRANSFER_END = auto()
    ALLREDUCE_START = auto()
    ALLREDUCE_END = auto()
    OPTIMIZER_START = auto()
    OPTIMIZER_END = auto()
    FAILURE = auto()
    RECOVERY = auto()


class Phase(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    BACKWARD_B = auto()  # activation gradient (ZeroBubble W-split)
    BACKWARD_W = auto()  # weight gradient (ZeroBubble W-split, deferrable)
    OPTIMIZER = auto()


class TaskStatus(Enum):
    WAITING = auto()
    READY = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def _is_half(self) -> bool:
        return self in (Precision.FP16, Precision.BF16)

    @property
    def data_scale(self) -> float:
        """Activation/gradient size multiplier (half-precision halves data)."""
        return 0.5 if self._is_half else 1.0

    @property
    def compute_speedup(self) -> float:
        """Compute speedup from tensor cores."""
        return 2.0 if self._is_half else 1.0

    @property
    def weight_memory_overhead(self) -> float:
        """FP32 master copy overhead (1.5x for mixed precision)."""
        return 1.5 if self._is_half else 1.0


class AllreduceAlgo(Enum):
    NAIVE = "naive"
    RING = "ring"


@dataclass(order=True)
class Event:
    time: float
    kind: EventKind = field(compare=False)
    payload: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class StageTask:
    microbatch_id: int
    stage_id: int
    phase: Phase
