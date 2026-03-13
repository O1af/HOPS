"""Pipeline model: stages, micro-batches, and forward/backward dataflow."""

from dataclasses import dataclass

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.scheduler import PipelineState, Scheduler
from hops.core.timing import TimingModel
from hops.core.types import Event, EventKind, Phase
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
                 rng: np.random.Generator | None = None):
        self.stages = {s.id: s for s in stages}
        self.stage_order = [s.id for s in stages]
        self._stage_index = {sid: i for i, sid in enumerate(self.stage_order)}
        self.engine = engine
        self.scheduler = scheduler
        self.collector = collector
        self.activation_size_mb = activation_size_mb
        self.rng = rng if rng is not None else np.random.default_rng()
        self.timing_model = TimingModel(topology, compute_model, self.rng)

        self._state = PipelineState(num_stages=len(stages), num_microbatches=0)
        self._batch_done_count = 0
        self._expected_done_count = 0
        self._total_mb_issued = 0  # running counter for globally unique MB IDs

        # Register event handlers
        engine.on(EventKind.COMPUTE_START, self._on_compute_start)
        engine.on(EventKind.COMPUTE_END, self._on_compute_end)
        engine.on(EventKind.TRANSFER_START, self._on_transfer_start)
        engine.on(EventKind.TRANSFER_END, self._on_transfer_end)

    def start_batch(self, num_microbatches: int) -> None:
        """Begin a training step with N micro-batches."""
        self._state = PipelineState.initialize(
            num_stages=len(self.stages),
            num_microbatches=num_microbatches,
        )
        self._batch_done_count = 0
        self._expected_done_count = num_microbatches
        self._batch_mb_offset = self._total_mb_issued
        self._total_mb_issued += num_microbatches
        self._issue_ready_tasks()

    @property
    def batch_complete(self) -> bool:
        return self._expected_done_count > 0 and self._batch_done_count >= self._expected_done_count

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
                self._state.mark_ready(stage_id, microbatch_id, Phase.BACKWARD)
        else:  # BACKWARD
            if stage_idx > 0:
                prev_stage_id = self.stage_order[stage_idx - 1]
                self._schedule_transfer(microbatch_id, phase, stage_id, prev_stage_id)
            else:
                self._batch_done_count += 1
                self.collector.record_microbatch_completion(global_microbatch_id, now)

        self._issue_ready_tasks()

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

        global_mb_id = mb_id + self._batch_mb_offset
        self.collector.record_transfer(
            global_mb_id, phase, event.payload["src_device"],
            event.payload["dst_device"], transfer_start, engine.now)

        self._advance_after_transfer_completion(
            stage_id=to_stage,
            microbatch_id=mb_id,
            phase=phase,
        )

    def _advance_after_transfer_completion(self, *, stage_id: int, microbatch_id: int,
                                           phase: Phase) -> None:
        """Release newly available work after a transfer completes."""
        self._state.mark_ready(stage_id, microbatch_id, phase)
        self._issue_ready_tasks()
