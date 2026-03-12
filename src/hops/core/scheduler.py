"""Scheduling policies for pipeline execution."""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

from hops.core.types import MicroBatch, Phase, StageTask


@dataclass
class PipelineState:
    num_stages: int
    num_microbatches: int
    completed: set[tuple[int, int, Phase]] = field(default_factory=set)
    in_flight: set[tuple[int, int, Phase]] = field(default_factory=set)
    now: float = 0.0
    # Incremental counters (maintained by Pipeline)
    fwd_completed: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    bwd_completed: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    in_flight_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def can_start_forward(self, stage: int, mb: int) -> bool:
        return stage == 0 or (stage - 1, mb, Phase.FORWARD) in self.completed

    def can_start_backward(self, stage: int, mb: int) -> bool:
        return stage == self.num_stages - 1 or (stage + 1, mb, Phase.BACKWARD) in self.completed


class Scheduler(ABC):
    @abstractmethod
    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        """Return tasks that should start now."""


class GPipeScheduler(Scheduler):
    """All forward passes first, then all backward passes."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        total_fwd = sum(s.fwd_completed[st] for st in range(s.num_stages))
        all_forwards_done = total_fwd == s.num_stages * s.num_microbatches

        if not all_forwards_done:
            for mb in range(s.num_microbatches):
                for stage in range(s.num_stages):
                    key = (stage, mb, Phase.FORWARD)
                    if key in s.completed or key in s.in_flight:
                        continue
                    if s.can_start_forward(stage, mb):
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.FORWARD))
            return tasks

        for mb in range(s.num_microbatches):
            for stage in reversed(range(s.num_stages)):
                key = (stage, mb, Phase.BACKWARD)
                if key in s.completed or key in s.in_flight:
                    continue
                if s.can_start_backward(stage, mb):
                    tasks.append(StageTask(MicroBatch(mb), stage, Phase.BACKWARD))
        return tasks


class OneFOneBScheduler(Scheduler):
    """1F1B: warmup forwards, then steady-state alternating forward/backward."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        for stage in range(s.num_stages):
            fwd_done = s.fwd_completed[stage]
            bwd_done = s.bwd_completed[stage]

            if s.in_flight_count[stage] > 0:
                continue

            warmup_limit = s.num_stages - stage

            # Prefer backward if we have pending backwards and past warmup
            if fwd_done > bwd_done and fwd_done >= warmup_limit:
                mb = bwd_done
                key = (stage, mb, Phase.BACKWARD)
                if key not in s.completed and key not in s.in_flight:
                    if s.can_start_backward(stage, mb):
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.BACKWARD))
                        continue

            # Issue next forward if available
            if fwd_done < s.num_microbatches:
                mb = fwd_done
                key = (stage, mb, Phase.FORWARD)
                if key not in s.completed and key not in s.in_flight:
                    if s.can_start_forward(stage, mb):
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.FORWARD))
                        continue

            # Fallback: issue backward if forward is done
            if bwd_done < s.num_microbatches and fwd_done > bwd_done:
                mb = bwd_done
                key = (stage, mb, Phase.BACKWARD)
                if key not in s.completed and key not in s.in_flight:
                    if s.can_start_backward(stage, mb):
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.BACKWARD))

        return tasks


def make_scheduler(config: dict) -> Scheduler:
    """Factory function to create a scheduler from config."""
    policy = config["policy"]
    schedulers = {
        "gpipe": GPipeScheduler,
        "1f1b": OneFOneBScheduler,
    }
    if policy not in schedulers:
        raise ValueError(f"Unknown scheduler policy: {policy}. Options: {list(schedulers)}")
    return schedulers[policy]()
