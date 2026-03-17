"""Pipeline model: stages, micro-batches, and forward/backward dataflow."""

from dataclasses import dataclass

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.scheduler import PipelineState, Scheduler, ZeroBubbleScheduler
from hops.core.timing import TimingModel
from hops.core.types import Event, EventKind, Phase
from hops.latency.distributions import Distribution
from hops.metrics.collector import MetricsCollector


@dataclass
class Stage:
    id: int
    device_id: str


class Pipeline:
    """Orchestrates micro-batch flow through pipeline stages via events."""

    def __init__(self, stages: list[Stage], engine: EventEngine,
                 topology, compute_model,
                 scheduler: Scheduler, collector: MetricsCollector,
                 activation_size_mb: float = 50.0,
                 rng: np.random.Generator | None = None,
                 optimizer_latency: Distribution | None = None,
                 gradient_size_mb: float = 0.0,
                 stage_memory_mb: dict[int, float] | None = None,
                 gradient_accumulation_steps: int = 1,
                 precision: str = "fp32",
                 allreduce_algo: str = "naive"):
        self.stages = {s.id: s for s in stages}
        self.stage_order = [s.id for s in stages]
        self._stage_index = {sid: i for i, sid in enumerate(self.stage_order)}
        self.engine = engine
        self.topology = topology
        self.scheduler = scheduler
        self.collector = collector
        self.rng = rng if rng is not None else np.random.default_rng()
        self.timing_model = TimingModel(topology, compute_model, self.rng)

        # Precision scaling
        self.precision = precision
        scale = 0.5 if precision in ("fp16", "bf16") else 1.0
        self.activation_size_mb = activation_size_mb * scale
        self.gradient_size_mb = gradient_size_mb * scale

        # Optimizer config
        self.optimizer_latency = optimizer_latency
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.allreduce_algo = allreduce_algo

        # ZeroBubble W-split detection
        self._use_w_split = isinstance(scheduler, ZeroBubbleScheduler)

        # Memory tracking
        self.stage_memory_mb = stage_memory_mb or {}
        for stage_id, mem in self.stage_memory_mb.items():
            device = self.topology.device(self.stages[stage_id].device_id)
            overhead = 1.5 if precision in ("fp16", "bf16") else 1.0
            device.allocate(mem * overhead)

        self._state = PipelineState(num_stages=len(stages), num_microbatches=0)
        self._batch_done_count = 0
        self._expected_done_count = 0
        self._total_mb_issued = 0
        self._optimizer_pending = 0
        self._allreduce_pending = 0
        self._in_optimizer_phase = False
        self._accumulation_count = 0
        self._is_optimizer_batch = False
        self._w_tasks_remaining = 0

        # Ring all-reduce state
        self._ring_devices: list[str] = []
        self._ring_chunk_size = 0.0
        self._ring_total_rounds = 0
        self._ring_current_round = 0

        # Register event handlers
        engine.on(EventKind.COMPUTE_START, self._on_compute_start)
        engine.on(EventKind.COMPUTE_END, self._on_compute_end)
        engine.on(EventKind.TRANSFER_START, self._on_transfer_start)
        engine.on(EventKind.TRANSFER_END, self._on_transfer_end)
        engine.on(EventKind.ALLREDUCE_START, self._on_allreduce_start)
        engine.on(EventKind.ALLREDUCE_END, self._on_allreduce_end)
        engine.on(EventKind.OPTIMIZER_START, self._on_optimizer_start)
        engine.on(EventKind.OPTIMIZER_END, self._on_optimizer_end)

    def start_batch(self, num_microbatches: int) -> None:
        """Begin a training step with N micro-batches."""
        self._state = PipelineState.initialize(
            num_stages=len(self.stages),
            num_microbatches=num_microbatches,
            use_w_split=self._use_w_split,
        )
        self._batch_done_count = 0
        self._expected_done_count = num_microbatches
        self._batch_mb_offset = self._total_mb_issued
        self._total_mb_issued += num_microbatches
        self._optimizer_pending = 0
        self._allreduce_pending = 0
        self._in_optimizer_phase = False
        self._w_tasks_remaining = (
            num_microbatches * len(self.stages) if self._use_w_split else 0
        )

        # Gradient accumulation
        self._accumulation_count += 1
        self._is_optimizer_batch = (
            self.optimizer_latency is not None
            and self._accumulation_count >= self.gradient_accumulation_steps
        )

        self._issue_ready_tasks()

    @property
    def batch_complete(self) -> bool:
        if self._expected_done_count <= 0:
            return False
        if self._batch_done_count < self._expected_done_count:
            return False
        if self._use_w_split and self._w_tasks_remaining > 0:
            return False
        if self._is_optimizer_batch:
            return (self._in_optimizer_phase
                    and self._optimizer_pending == 0
                    and self._allreduce_pending == 0)
        return True

    def set_failure_engine(self, failure_engine) -> None:
        self.timing_model.set_failure_engine(failure_engine)

    def _issue_ready_tasks(self) -> None:
        """Ask the scheduler what to run and schedule COMPUTE_START events."""
        tasks = self.scheduler.next_tasks(self._state)
        for task in tasks:
            if not self._state.is_task_ready(task.stage_id, task.microbatch_id, task.phase):
                continue
            self._state.reserve_task(task.stage_id, task.microbatch_id, task.phase)
            self.collector.record_in_flight(
                task.stage_id, self.engine.now, self._state.stage_in_flight_count(task.stage_id))
            self.engine.schedule(Event(
                time=self.engine.now,
                kind=EventKind.COMPUTE_START,
                payload={"stage_id": task.stage_id,
                         "microbatch_id": task.microbatch_id,
                         "phase": task.phase},
            ))

    def _on_compute_start(self, event: Event, engine: EventEngine) -> None:
        stage_id = event.payload["stage_id"]
        mb_id = event.payload["microbatch_id"]
        phase = event.payload["phase"]
        device_id = self.stages[stage_id].device_id
        start_time, end_time = self.timing_model.reserve_compute(
            now=engine.now,
            stage_id=stage_id,
            device_id=device_id,
            phase=phase,
        )
        if start_time > engine.now:
            engine.schedule(Event(
                time=start_time,
                kind=EventKind.COMPUTE_START,
                payload=event.payload,
            ))
            return

        self._state.begin_compute(stage_id, mb_id, phase)

        engine.schedule(Event(
            time=end_time,
            kind=EventKind.COMPUTE_END,
            payload={"stage_id": stage_id, "microbatch_id": mb_id,
                     "phase": phase, "compute_start": start_time},
        ))

    def _on_compute_end(self, event: Event, engine: EventEngine) -> None:
        stage_id = event.payload["stage_id"]
        mb_id = event.payload["microbatch_id"]
        phase = event.payload["phase"]
        start_time = event.payload["compute_start"]
        device_id = self.stages[stage_id].device_id

        # Memory tracking: forward produces activations, backward consumes them
        if phase == Phase.FORWARD:
            device = self.topology.device(device_id)
            device.allocate(self.activation_size_mb)
            self.collector.record_peak_memory(device_id, device.peak_memory_mb)
        elif phase in (Phase.BACKWARD, Phase.BACKWARD_B):
            device = self.topology.device(device_id)
            device.free(self.activation_size_mb)

        # Record metrics with globally unique MB ID
        global_mb_id = mb_id + self._batch_mb_offset
        self.collector.record_compute(stage_id, global_mb_id, phase, device_id,
                                      start_time, engine.now)

        self._state.finish_task(stage_id, mb_id, phase)
        self.collector.record_in_flight(stage_id, engine.now, self._state.stage_in_flight_count(stage_id))
        self._advance_after_task_completion(
            stage_id=stage_id,
            microbatch_id=mb_id,
            phase=phase,
            global_microbatch_id=global_mb_id,
            now=engine.now,
        )

    def _advance_after_task_completion(self, *, stage_id: int, microbatch_id: int,
                                       phase: Phase, global_microbatch_id: int,
                                       now: float) -> None:
        """Apply logical dependency transitions after a compute finishes."""
        stage_idx = self._stage_index[stage_id]

        if phase == Phase.FORWARD:
            if stage_idx < len(self.stage_order) - 1:
                next_stage_id = self.stage_order[stage_idx + 1]
                self._schedule_transfer(microbatch_id, phase, stage_id, next_stage_id)
            else:
                # Last stage: start backward
                bwd_phase = Phase.BACKWARD_B if self._use_w_split else Phase.BACKWARD
                self._state.mark_ready(stage_id, microbatch_id, bwd_phase)

        elif phase == Phase.BACKWARD:
            if stage_idx > 0:
                prev_stage_id = self.stage_order[stage_idx - 1]
                self._schedule_transfer(microbatch_id, phase, stage_id, prev_stage_id)
            else:
                self._batch_done_count += 1
                self.collector.record_microbatch_completion(global_microbatch_id, now)
                self._check_optimizer_trigger()

        elif phase == Phase.BACKWARD_B:
            # Mark W task as ready (deferrable)
            self._state.mark_ready(stage_id, microbatch_id, Phase.BACKWARD_W)
            # Transfer activation gradient to previous stage
            if stage_idx > 0:
                prev_stage_id = self.stage_order[stage_idx - 1]
                self._schedule_transfer(microbatch_id, phase, stage_id, prev_stage_id)
            else:
                self._batch_done_count += 1
                self.collector.record_microbatch_completion(global_microbatch_id, now)
                self._check_optimizer_trigger()

        elif phase == Phase.BACKWARD_W:
            self._w_tasks_remaining -= 1
            self._check_optimizer_trigger()

        self._issue_ready_tasks()

    def _check_optimizer_trigger(self) -> None:
        """Start optimizer phase if all conditions are met."""
        if self._batch_done_count < self._expected_done_count:
            return
        if self._use_w_split and self._w_tasks_remaining > 0:
            return
        if not self._is_optimizer_batch:
            return
        if self._in_optimizer_phase:
            return
        self._accumulation_count = 0
        self._start_optimizer_phase()

    def _start_optimizer_phase(self) -> None:
        """Kick off all-reduce followed by optimizer compute on each device."""
        self._in_optimizer_phase = True
        device_ids = [self.stages[sid].device_id for sid in self.stage_order]
        unique_devices = list(dict.fromkeys(device_ids))

        if self.gradient_size_mb > 0 and len(unique_devices) > 1:
            if self.allreduce_algo == "ring":
                self._start_ring_allreduce(unique_devices)
            else:
                self._start_naive_allreduce(unique_devices)
        else:
            self._start_optimizer_compute()

    def _start_naive_allreduce(self, unique_devices: list[str]) -> None:
        """Adjacent pairwise all-reduce (original algorithm)."""
        pairs = []
        for i in range(len(unique_devices) - 1):
            pairs.append((unique_devices[i], unique_devices[i + 1]))
            pairs.append((unique_devices[i + 1], unique_devices[i]))
        for src, dst in pairs:
            try:
                self.topology.link(src, dst)
            except KeyError:
                raise ValueError(
                    f"Optimizer all-reduce requires a link from {src!r} to {dst!r}, "
                    f"but none exists in the topology"
                )
        self._allreduce_pending = len(pairs)
        for src, dst in pairs:
            self.engine.schedule(Event(
                time=self.engine.now,
                kind=EventKind.ALLREDUCE_START,
                payload={"src_device": src, "dst_device": dst,
                         "size_mb": self.gradient_size_mb},
            ))

    def _start_ring_allreduce(self, unique_devices: list[str]) -> None:
        """Ring all-reduce: 2*(N-1) rounds, N parallel transfers per round."""
        N = len(unique_devices)
        # Validate ring topology
        for i in range(N):
            src, dst = unique_devices[i], unique_devices[(i + 1) % N]
            try:
                self.topology.link(src, dst)
            except KeyError:
                raise ValueError(
                    f"Ring all-reduce requires a link from {src!r} to {dst!r}. "
                    f"Add the link or use allreduce_algo='naive'."
                )

        self._ring_devices = unique_devices
        self._ring_chunk_size = self.gradient_size_mb / N
        self._ring_total_rounds = 2 * (N - 1)
        self._ring_current_round = 0
        self._start_ring_round()

    def _start_ring_round(self) -> None:
        """Launch one round of ring all-reduce (N parallel transfers)."""
        N = len(self._ring_devices)
        self._allreduce_pending = N
        for i in range(N):
            src = self._ring_devices[i]
            dst = self._ring_devices[(i + 1) % N]
            self.engine.schedule(Event(
                time=self.engine.now,
                kind=EventKind.ALLREDUCE_START,
                payload={"src_device": src, "dst_device": dst,
                         "size_mb": self._ring_chunk_size},
            ))

    def _on_allreduce_start(self, event: Event, engine: EventEngine) -> None:
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]
        size_mb = event.payload["size_mb"]
        start_time, end_time = self.timing_model.reserve_transfer(
            now=engine.now, src_device=src, dst_device=dst, size_mb=size_mb,
        )
        if start_time > engine.now:
            engine.schedule(Event(time=start_time, kind=EventKind.ALLREDUCE_START,
                                  payload=event.payload))
            return

        engine.schedule(Event(
            time=end_time,
            kind=EventKind.ALLREDUCE_END,
            payload={**event.payload, "transfer_start": start_time},
        ))

    def _on_allreduce_end(self, event: Event, engine: EventEngine) -> None:
        transfer_start = event.payload["transfer_start"]
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]

        self.timing_model.release_transfer(src, dst)

        self.collector.record_transfer(
            None, Phase.OPTIMIZER, src, dst, transfer_start, engine.now)

        self._allreduce_pending -= 1
        if self._allreduce_pending == 0:
            if self.allreduce_algo == "ring":
                self._ring_current_round += 1
                if self._ring_current_round < self._ring_total_rounds:
                    self._start_ring_round()
                    return
            self._start_optimizer_compute()

    def _start_optimizer_compute(self) -> None:
        """Schedule optimizer weight update on each device."""
        device_ids = [self.stages[sid].device_id for sid in self.stage_order]
        unique_devices = list(dict.fromkeys(device_ids))
        self._optimizer_pending = len(unique_devices)
        for device_id in unique_devices:
            self.engine.schedule(Event(
                time=self.engine.now,
                kind=EventKind.OPTIMIZER_START,
                payload={"device_id": device_id},
            ))

    def _on_optimizer_start(self, event: Event, engine: EventEngine) -> None:
        device_id = event.payload["device_id"]
        duration = event.payload.get("duration")
        if duration is None:
            duration = self.optimizer_latency.sample(self.rng)

        start_time, end_time = self.timing_model.reserve_device(
            now=engine.now, device_id=device_id, duration=duration,
        )
        if start_time > engine.now:
            engine.schedule(Event(
                time=start_time, kind=EventKind.OPTIMIZER_START,
                payload={"device_id": device_id, "duration": duration},
            ))
            return

        engine.schedule(Event(
            time=end_time,
            kind=EventKind.OPTIMIZER_END,
            payload={"device_id": device_id, "compute_start": start_time},
        ))

    def _on_optimizer_end(self, event: Event, engine: EventEngine) -> None:
        device_id = event.payload["device_id"]
        start_time = event.payload["compute_start"]
        stage_id = next(sid for sid in self.stage_order
                        if self.stages[sid].device_id == device_id)
        self.collector.record_compute(stage_id, None, Phase.OPTIMIZER, device_id,
                                      start_time, engine.now)
        self._optimizer_pending -= 1

    def _schedule_transfer(self, mb_id: int, phase: Phase,
                           from_stage: int, to_stage: int) -> None:
        src_device = self.stages[from_stage].device_id
        dst_device = self.stages[to_stage].device_id

        self.engine.schedule(Event(
            time=self.engine.now,
            kind=EventKind.TRANSFER_START,
            payload={"microbatch_id": mb_id, "phase": phase,
                     "from_stage": from_stage, "to_stage": to_stage,
                     "src_device": src_device, "dst_device": dst_device},
        ))

    def _on_transfer_start(self, event: Event, engine: EventEngine) -> None:
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]
        start_time, end_time = self.timing_model.reserve_transfer(
            now=engine.now,
            src_device=src,
            dst_device=dst,
            size_mb=self.activation_size_mb,
        )
        if start_time > engine.now:
            engine.schedule(Event(
                time=start_time,
                kind=EventKind.TRANSFER_START,
                payload=event.payload,
            ))
            return

        engine.schedule(Event(
            time=end_time,
            kind=EventKind.TRANSFER_END,
            payload={**event.payload, "transfer_start": start_time},
        ))

    def _on_transfer_end(self, event: Event, engine: EventEngine) -> None:
        mb_id = event.payload["microbatch_id"]
        phase = event.payload["phase"]
        to_stage = event.payload["to_stage"]
        transfer_start = event.payload["transfer_start"]
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]

        self.timing_model.release_transfer(src, dst)

        global_mb_id = mb_id + self._batch_mb_offset
        self.collector.record_transfer(
            global_mb_id, phase, src, dst, transfer_start, engine.now)

        self._advance_after_transfer_completion(
            stage_id=to_stage,
            microbatch_id=mb_id,
            phase=phase,
        )

    def _advance_after_transfer_completion(self, *, stage_id: int, microbatch_id: int,
                                           phase: Phase) -> None:
        """Release newly available work after a transfer completes."""
        if phase == Phase.FORWARD:
            # Forward transfer arrived: mark forward at destination ready
            self._state.mark_ready(stage_id, microbatch_id, Phase.FORWARD)
        elif phase == Phase.BACKWARD_B:
            # Activation gradient arrived: mark BACKWARD_B at destination ready
            self._state.mark_ready(stage_id, microbatch_id, Phase.BACKWARD_B)
        else:
            # BACKWARD transfer: mark backward at destination ready
            self._state.mark_ready(stage_id, microbatch_id, phase)
        self._issue_ready_tasks()
