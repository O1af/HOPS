"""Scheduling policies for pipeline execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from hops.core.types import Phase, StageTask, TaskStatus


@dataclass
class PipelineState:
    num_stages: int
    num_microbatches: int
    task_states: dict[tuple[int, int, Phase], TaskStatus] = field(default_factory=dict)

    @classmethod
    def initialize(cls, num_stages: int, num_microbatches: int,
                   use_w_split: bool = False) -> "PipelineState":
        task_states = {}
        for mb in range(num_microbatches):
            for stage in range(num_stages):
                task_states[(stage, mb, Phase.FORWARD)] = TaskStatus.WAITING
                if use_w_split:
                    task_states[(stage, mb, Phase.BACKWARD_B)] = TaskStatus.WAITING
                    task_states[(stage, mb, Phase.BACKWARD_W)] = TaskStatus.WAITING
                else:
                    task_states[(stage, mb, Phase.BACKWARD)] = TaskStatus.WAITING
            task_states[(0, mb, Phase.FORWARD)] = TaskStatus.READY
        return cls(
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            task_states=task_states,
        )

    def is_task_ready(self, stage_id: int, microbatch_id: int, phase: Phase) -> bool:
        key = (stage_id, microbatch_id, phase)
        return key in self.task_states and self.task_states[key] == TaskStatus.READY

    def mark_ready(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if key in self.task_states and self.task_states[key] != TaskStatus.COMPLETED:
            self.task_states[key] = TaskStatus.READY

    def reserve_task(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if self.task_states[key] != TaskStatus.READY:
            raise ValueError(f"Cannot reserve task {key} from state {self.task_states[key]}")
        self.task_states[key] = TaskStatus.SCHEDULED

    def begin_compute(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if self.task_states[key] != TaskStatus.SCHEDULED:
            raise ValueError(f"Cannot start compute for task {key} from state {self.task_states[key]}")
        self.task_states[key] = TaskStatus.RUNNING

    def finish_task(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if self.task_states[key] != TaskStatus.RUNNING:
            raise ValueError(f"Cannot finish task {key} from state {self.task_states[key]}")
        self.task_states[key] = TaskStatus.COMPLETED

    def stage_is_busy(self, stage_id: int) -> bool:
        return any(
            status == TaskStatus.RUNNING and task_stage == stage_id
            for (task_stage, _, _), status in self.task_states.items()
        )

    def stage_in_flight_count(self, stage_id: int) -> int:
        return sum(
            1
            for (task_stage, _, _), status in self.task_states.items()
            if task_stage == stage_id and status in {TaskStatus.SCHEDULED, TaskStatus.RUNNING}
        )

    def completed_count(self, stage_id: int, phase: Phase) -> int:
        return sum(
            1
            for (task_stage, _, task_phase), status in self.task_states.items()
            if task_stage == stage_id and task_phase == phase and status == TaskStatus.COMPLETED
        )

    def all_forwards_completed(self) -> bool:
        total_fwd = sum(
            1
            for (_, _, phase), status in self.task_states.items()
            if phase == Phase.FORWARD and status == TaskStatus.COMPLETED
        )
        return total_fwd == self.num_stages * self.num_microbatches


class Scheduler(ABC):
    @abstractmethod
    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        """Return tasks that should start now."""


class GPipeScheduler(Scheduler):
    """All forward passes first, then all backward passes."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        if not s.all_forwards_completed():
            for mb in range(s.num_microbatches):
                for stage in range(s.num_stages):
                    if s.is_task_ready(stage, mb, Phase.FORWARD):
                        tasks.append(StageTask(mb, stage, Phase.FORWARD))
            return tasks

        for mb in range(s.num_microbatches):
            for stage in reversed(range(s.num_stages)):
                if s.is_task_ready(stage, mb, Phase.BACKWARD):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD))
        return tasks


class OneFOneBScheduler(Scheduler):
    """1F1B: warmup forwards, then steady-state alternating forward/backward."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        for stage in range(s.num_stages):
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            bwd_done = s.completed_count(stage, Phase.BACKWARD)

            if s.stage_is_busy(stage):
                continue

            warmup_limit = s.num_stages - stage

            # Prefer backward if we have pending backwards and past warmup
            if fwd_done > bwd_done and fwd_done >= warmup_limit:
                mb = bwd_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD))
                    continue

            # Issue next forward if available
            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            # Fallback: issue backward if forward is done
            if bwd_done < s.num_microbatches and fwd_done > bwd_done:
                mb = bwd_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD))

        return tasks


class ZeroBubbleScheduler(Scheduler):
    """ZeroBubble: splits backward into B (activation grad) and W (weight grad).

    W tasks have no downstream dependency and are deferred to fill bubbles.
    Three phases per stage:
      - Warmup:   issue (num_stages - stage_idx) forwards
      - Steady:   alternate F, B — W only runs when both F and B are blocked
      - Cooldown: finish remaining B tasks, then drain all deferred W tasks
    """

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue

            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)

            warmup_limit = s.num_stages - stage
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches

            # ── Cooldown: all F and B done → drain deferred W tasks ──
            if all_fwd_done and all_b_done:
                if w_done < b_done:
                    mb = w_done
                    if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                continue

            # ── Priority 1: BACKWARD_B — keeps the pipeline moving ──
            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            # ── Priority 2: FORWARD — feeds the pipeline ──
            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            # ── Bubble fill: neither F nor B is ready → run a deferred W ──
            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            # ── Fallback: try B if available ──
            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


def max_in_flight_count(policy: str, stage_idx: int,
                        num_stages: int, num_microbatches: int) -> int:
    """Upper bound on simultaneous activations stored at a stage."""
    if policy == "gpipe":
        return num_microbatches
    # 1F1B and ZeroBubble: bounded by pipeline depth
    return min(num_stages - stage_idx, num_microbatches)


_SCHEDULER_REGISTRY: dict[str, type[Scheduler]] = {
    "gpipe": GPipeScheduler,
    "1f1b": OneFOneBScheduler,
    "zero_bubble": ZeroBubbleScheduler,
}


def register_scheduler(name: str, cls: type[Scheduler]) -> None:
    """Register a custom scheduling policy by name."""
    if not isinstance(cls, type) or not issubclass(cls, Scheduler):
        raise TypeError(
            f"Expected a Scheduler subclass, got {cls!r}"
        )
    _SCHEDULER_REGISTRY[name] = cls


def make_scheduler(config: dict) -> Scheduler:
    """Factory function to create a scheduler from config."""
    policy = config["policy"]
    if policy not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler policy: {policy}. Options: {list(_SCHEDULER_REGISTRY)}")
    return _SCHEDULER_REGISTRY[policy]()
