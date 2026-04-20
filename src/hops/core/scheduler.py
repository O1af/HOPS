"""Scheduling policies for pipeline execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hops.core.types import Phase, StageTask, TaskStatus

if TYPE_CHECKING:
    from hops.config import AppConfig
    from hops.hardware.topology import Topology
    from hops.presets import PresetRegistry

_PER_LAYER_KERNEL_MS = 1.4


@dataclass
class PipelineState:
    num_stages: int
    num_microbatches: int
    task_states: dict[tuple[int, int, Phase], TaskStatus] = field(default_factory=dict)
    # O(1) counters maintained during state transitions
    _running_per_stage: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _in_flight_per_stage: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    _completed: dict[tuple[int, Phase], int] = field(default_factory=lambda: defaultdict(int))
    _total_fwd_completed: int = 0

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
        if key not in self.task_states:
            return
        status = self.task_states[key]
        if status == TaskStatus.WAITING:
            self.task_states[key] = TaskStatus.READY
        elif status in (TaskStatus.SCHEDULED, TaskStatus.RUNNING, TaskStatus.COMPLETED):
            return

    def reserve_task(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if self.task_states[key] != TaskStatus.READY:
            raise ValueError(f"Cannot reserve task {key} from state {self.task_states[key]}")
        self.task_states[key] = TaskStatus.SCHEDULED
        self._in_flight_per_stage[stage_id] += 1

    def begin_compute(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if self.task_states[key] != TaskStatus.SCHEDULED:
            raise ValueError(f"Cannot start compute for task {key} from state {self.task_states[key]}")
        self.task_states[key] = TaskStatus.RUNNING
        self._running_per_stage[stage_id] += 1

    def finish_task(self, stage_id: int, microbatch_id: int, phase: Phase) -> None:
        key = (stage_id, microbatch_id, phase)
        if self.task_states[key] != TaskStatus.RUNNING:
            raise ValueError(f"Cannot finish task {key} from state {self.task_states[key]}")
        self.task_states[key] = TaskStatus.COMPLETED
        self._running_per_stage[stage_id] -= 1
        self._in_flight_per_stage[stage_id] -= 1
        self._completed[(stage_id, phase)] += 1
        if phase == Phase.FORWARD:
            self._total_fwd_completed += 1

    def stage_is_busy(self, stage_id: int) -> bool:
        return self._running_per_stage[stage_id] > 0

    def stage_in_flight_count(self, stage_id: int) -> int:
        return self._in_flight_per_stage[stage_id]

    def completed_count(self, stage_id: int, phase: Phase) -> int:
        return self._completed[(stage_id, phase)]

    def all_forwards_completed(self) -> bool:
        return self._total_fwd_completed == self.num_stages * self.num_microbatches


class Scheduler(ABC):
    uses_w_split: bool = False

    @abstractmethod
    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        """Return tasks that should start now."""


class GPipeScheduler(Scheduler):
    """All forward passes first, then all backward passes."""

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks = []
        s = state

        if not s.all_forwards_completed():
            for stage in range(s.num_stages):
                mb = s.completed_count(stage, Phase.FORWARD)
                if mb < s.num_microbatches and s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
            return tasks

        for stage in reversed(range(s.num_stages)):
            mb = s.completed_count(stage, Phase.BACKWARD)
            if mb < s.num_microbatches and s.is_task_ready(stage, mb, Phase.BACKWARD):
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
    uses_w_split: bool = True

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


def _estimate_layer_count(tflop: float, model: Any) -> float:
    if model is None or tflop <= 0:
        return 0.0
    per_layer_tflop = 24.0 * model.seq_len * (model.hidden_dim**2) / 1e12
    if per_layer_tflop <= 0:
        return 0.0
    return tflop / per_layer_tflop


def _analytical_forward_ms(
    *,
    tflop: float,
    memory_mb: float,
    device_flops: float,
    memory_bw_gbps: float,
    efficiency_compute: float,
    efficiency_memory: float,
    compute_scale: float,
    memory_bw_scale: float,
    memory_latency_us: float,
    launch_overhead_ms: float,
    layer_count: float,
    model: Any,
) -> float:
    effective_flops = device_flops * efficiency_compute
    compute_ms = 1000.0 * tflop / effective_flops * compute_scale
    memory_ms = 0.0
    if memory_mb > 0 and memory_bw_gbps > 0:
        effective_bw = memory_bw_gbps * efficiency_memory * memory_bw_scale
        memory_ms = (memory_mb * 8.0) / effective_bw + memory_latency_us / 1000.0
    kernel_floor_ms = _PER_LAYER_KERNEL_MS * layer_count
    return max(compute_ms, memory_ms, kernel_floor_ms) + launch_overhead_ms


def build_stage_compute_weights(
    app: AppConfig,
    topology: Topology,
    registry: PresetRegistry,
) -> list[float]:
    from hops.config import StageConfig

    precision_speedup = app.pipeline.precision.compute_speedup
    model = app.pipeline.model
    b_frac = (
        app.pipeline.backward_split.activation_grad_fraction
        if app.pipeline.backward_split.enabled
        else 0.5
    )
    bf = app.pipeline.backward_factor
    b_scale = bf * b_frac
    w_scale = bf * (1.0 - b_frac)
    overrides = {o.id: o for o in app.overrides.devices}
    spec_by_id = {d.id: d for d in app.hardware.devices}
    weights: list[float] = []
    for stage in sorted(app.pipeline.stages, key=lambda s: s.id):
        assert isinstance(stage, StageConfig)
        dev_spec = spec_by_id[stage.device]
        preset = registry.device(dev_spec.preset)
        override = overrides.get(dev_spec.id)
        flops = (
            override.flops_tflops
            if override and override.flops_tflops is not None
            else preset.flops_tflops
        )
        mbw = (
            override.memory_bandwidth_gbps
            if override and override.memory_bandwidth_gbps is not None
            else preset.memory_bandwidth_gbps
        )
        launch = (
            override.launch_overhead_ms
            if override and override.launch_overhead_ms is not None
            else preset.launch_overhead_ms
        )
        if stage.compute_mode != "analytical" or stage.analytical is None:
            weights.append(1.0)
            continue
        an = stage.analytical
        penalty = topology.stage_locality_penalty(
            device_id=stage.device,
            memory_placement=stage.memory_placement,
        )
        compute_scale = penalty.compute_scale / precision_speedup
        lc = _estimate_layer_count(an.tflop, model)
        fwd = _analytical_forward_ms(
            tflop=an.tflop,
            memory_mb=an.memory_mb,
            device_flops=flops,
            memory_bw_gbps=mbw or 1.0,
            efficiency_compute=an.efficiency_compute,
            efficiency_memory=an.efficiency_memory,
            compute_scale=compute_scale,
            memory_bw_scale=penalty.memory_bandwidth_scale,
            memory_latency_us=penalty.memory_latency_us,
            launch_overhead_ms=launch if launch is not None else 1.5,
            layer_count=lc,
            model=model,
        )
        weights.append(fwd * (1.0 + b_scale + w_scale))
    return weights


class HeterogeneousHopsScheduler(Scheduler):
    uses_w_split: bool = True

    def __init__(self, stage_weights: list[float]) -> None:
        self._w = list(stage_weights)
        if not self._w:
            raise ValueError("stage_weights must be non-empty")
        med = sorted(self._w)[len(self._w) // 2]
        med = med if med > 1e-9 else 1.0
        self._rel = [max(1e-9, x) / med for x in self._w]
        self._crit = int(max(range(len(self._rel)), key=lambda i: self._rel[i]))

    @classmethod
    def from_app(
        cls,
        app: AppConfig,
        topology: Topology,
        registry: PresetRegistry,
    ) -> HeterogeneousHopsScheduler:
        return cls(build_stage_compute_weights(app, topology, registry))

    def _warmup_limit(self, stage: int, num_microbatches: int) -> int:
        s = len(self._rel)
        base = s - stage
        if stage >= s - 1:
            return min(base, num_microbatches)
        rs = self._rel[stage]
        rd = self._rel[stage + 1]
        ratio = rd / rs if rs > 1e-12 else 1.0
        extra = max(0, int(round(ratio - 1.0)))
        return min(num_microbatches, base + extra)

    def _defer_w(self, stage: int) -> bool:
        r = self._rel[stage]
        if stage == self._crit:
            return False
        return r < 1.0 - 1e-9

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue

            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)

            warmup_limit = self._warmup_limit(stage, s.num_microbatches)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches

            if all_fwd_done and all_b_done:
                if w_done < b_done:
                    mb = w_done
                    if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                continue

            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    if self._defer_w(stage):
                        pass
                    else:
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        if not tasks:
            for stage in range(s.num_stages):
                if s.stage_is_busy(stage):
                    continue
                fwd_done = s.completed_count(stage, Phase.FORWARD)
                b_done = s.completed_count(stage, Phase.BACKWARD_B)
                w_done = s.completed_count(stage, Phase.BACKWARD_W)
                all_fwd_done = fwd_done >= s.num_microbatches
                all_b_done = b_done >= s.num_microbatches
                if all_fwd_done and all_b_done:
                    continue
                if w_done < b_done and self._defer_w(stage):
                    mb = w_done
                    if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                        break

        return tasks


def max_in_flight_count(policy: str, stage_idx: int,
                        num_stages: int, num_microbatches: int) -> int:
    """Upper bound on simultaneous activations stored at a stage."""
    if policy == "gpipe":
        return num_microbatches
    # 1F1B, ZeroBubble, heterogeneous_hops: bounded by pipeline depth
    return min(num_stages - stage_idx, num_microbatches)


_SCHEDULER_REGISTRY: dict[str, type[Scheduler]] = {
    "gpipe": GPipeScheduler,
    "1f1b": OneFOneBScheduler,
    "zero_bubble": ZeroBubbleScheduler,
    "heterogeneous_hops": HeterogeneousHopsScheduler,
}


def register_scheduler(name: str, cls: type[Scheduler]) -> None:
    """Register a custom scheduling policy by name."""
    if not isinstance(cls, type) or not issubclass(cls, Scheduler):
        raise TypeError(
            f"Expected a Scheduler subclass, got {cls!r}"
        )
    _SCHEDULER_REGISTRY[name] = cls


def make_scheduler(
    config: dict,
    *,
    app: AppConfig | None = None,
    topology: Topology | None = None,
    registry: PresetRegistry | None = None,
) -> Scheduler:
    """Factory function to create a scheduler from config."""
    policy = config["policy"]
    if policy not in _SCHEDULER_REGISTRY:
        raise ValueError(f"Unknown scheduler policy: {policy}. Options: {list(_SCHEDULER_REGISTRY)}")
    cls = _SCHEDULER_REGISTRY[policy]
    if cls is HeterogeneousHopsScheduler:
        if app is None or topology is None or registry is None:
            raise ValueError(
                "heterogeneous_hops requires app, topology, and registry "
                "arguments to make_scheduler()"
            )
        return HeterogeneousHopsScheduler.from_app(app, topology, registry)
    return cls()
