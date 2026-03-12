"""Scheduling policies for pipeline execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hops.core.types import MicroBatch, Phase, StageTask


@dataclass
class PipelineState:
    num_stages: int
    num_microbatches: int
    completed: set[tuple[int, int, Phase]] = field(default_factory=set)  # (stage, mb, phase)
    in_flight: set[tuple[int, int, Phase]] = field(default_factory=set)
    now: float = 0.0


class Scheduler(ABC):
    @abstractmethod
    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        """Return tasks that should start now."""


class GPipeScheduler(Scheduler):
    """All forward passes first, then all backward passes."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        # Phase 1: issue all forwards
        all_forwards_done = all(
            (stage, mb, Phase.FORWARD) in s.completed
            for stage in range(s.num_stages)
            for mb in range(s.num_microbatches)
        )

        if not all_forwards_done:
            # Issue forward passes stage by stage
            for mb in range(s.num_microbatches):
                for stage in range(s.num_stages):
                    key = (stage, mb, Phase.FORWARD)
                    if key in s.completed or key in s.in_flight:
                        continue
                    # Can start if previous stage is done (or stage 0)
                    if stage == 0 or (stage - 1, mb, Phase.FORWARD) in s.completed:
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.FORWARD))
            return tasks

        # Phase 2: issue backward passes (reverse stage order)
        for mb in range(s.num_microbatches):
            for stage in reversed(range(s.num_stages)):
                key = (stage, mb, Phase.BACKWARD)
                if key in s.completed or key in s.in_flight:
                    continue
                # Can start if next stage backward is done (or last stage)
                if stage == s.num_stages - 1 or (stage + 1, mb, Phase.BACKWARD) in s.completed:
                    tasks.append(StageTask(MicroBatch(mb), stage, Phase.BACKWARD))
        return tasks


class OneFOneBScheduler(Scheduler):
    """1F1B: warmup forwards, then steady-state alternating forward/backward."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        for stage in range(s.num_stages):
            # Count how many forwards and backwards this stage has completed
            fwd_done = sum(1 for mb in range(s.num_microbatches)
                          if (stage, mb, Phase.FORWARD) in s.completed)
            bwd_done = sum(1 for mb in range(s.num_microbatches)
                          if (stage, mb, Phase.BACKWARD) in s.completed)
            in_flight_here = any((stage, mb, p) in s.in_flight
                                for mb in range(s.num_microbatches)
                                for p in Phase)

            if in_flight_here:
                continue

            # Warmup: allow up to (num_stages - stage) forwards before first backward
            warmup_limit = s.num_stages - stage

            # Prefer backward if we have pending backwards and past warmup
            if fwd_done > bwd_done and fwd_done >= warmup_limit:
                # Issue next backward
                mb = bwd_done
                key = (stage, mb, Phase.BACKWARD)
                if key not in s.completed and key not in s.in_flight:
                    if stage == s.num_stages - 1 or (stage + 1, mb, Phase.BACKWARD) in s.completed:
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.BACKWARD))
                        continue

            # Issue next forward if available
            if fwd_done < s.num_microbatches:
                mb = fwd_done
                key = (stage, mb, Phase.FORWARD)
                if key not in s.completed and key not in s.in_flight:
                    if stage == 0 or (stage - 1, mb, Phase.FORWARD) in s.completed:
                        tasks.append(StageTask(MicroBatch(mb), stage, Phase.FORWARD))
                        continue

            # Fallback: issue backward if forward is done
            if bwd_done < s.num_microbatches and fwd_done > bwd_done:
                mb = bwd_done
                key = (stage, mb, Phase.BACKWARD)
                if key not in s.completed and key not in s.in_flight:
                    if stage == s.num_stages - 1 or (stage + 1, mb, Phase.BACKWARD) in s.completed:
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
