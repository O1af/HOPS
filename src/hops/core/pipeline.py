"""Pipeline model: stages, micro-batches, and forward/backward dataflow."""

from dataclasses import dataclass

from hops.core.event_engine import EventEngine
from hops.core.scheduler import PipelineState, Scheduler
from hops.core.types import Event, EventKind, MicroBatch, Phase, StageTask
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.metrics.collector import MetricsCollector


@dataclass
class Stage:
    id: int
    device_id: str
    num_layers: int = 1
    memory_mb: float = 0.0


class Pipeline:
    """Orchestrates micro-batch flow through pipeline stages via events."""

    def __init__(self, stages: list[Stage], engine: EventEngine,
                 topology: Topology, compute_model: ComputeModel,
                 scheduler: Scheduler, collector: MetricsCollector,
                 activation_size_mb: float = 50.0):
        self.stages = {s.id: s for s in stages}
        self.stage_order = [s.id for s in stages]
        self.engine = engine
        self.topology = topology
        self.compute_model = compute_model
        self.scheduler = scheduler
        self.collector = collector
        self.activation_size_mb = activation_size_mb

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
        self._state = PipelineState(
            num_stages=len(self.stages),
            num_microbatches=num_microbatches,
            now=self.engine.now,
        )
        self._batch_done_count = 0
        self._expected_done_count = num_microbatches
        self._batch_mb_offset = self._total_mb_issued
        self._total_mb_issued += num_microbatches
        self._issue_ready_tasks()

    def _issue_ready_tasks(self) -> None:
        """Ask the scheduler what to run and schedule COMPUTE_START events."""
        self._state.now = self.engine.now
        tasks = self.scheduler.next_tasks(self._state)
        for task in tasks:
            key = (task.stage_id, task.microbatch.id, task.phase)
            if key in self._state.in_flight:
                continue
            self._state.in_flight.add(key)
            stage = self.stages[task.stage_id]
            device = self.topology.device(stage.device_id)
            # Respect device busy_until
            start_time = max(self.engine.now, device.busy_until)
            self.engine.schedule(Event(
                time=start_time,
                kind=EventKind.COMPUTE_START,
                payload={"stage_id": task.stage_id,
                         "microbatch_id": task.microbatch.id,
                         "phase": task.phase},
            ))

    def _on_compute_start(self, event: Event, engine: EventEngine) -> None:
        stage_id = event.payload["stage_id"]
        mb_id = event.payload["microbatch_id"]
        phase = event.payload["phase"]
        device_id = self.stages[stage_id].device_id
        device = self.topology.device(device_id)

        # If device is still busy, defer this compute
        if device.busy_until > engine.now:
            engine.schedule(Event(
                time=device.busy_until,
                kind=EventKind.COMPUTE_START,
                payload=event.payload,
            ))
            return

        latency = self.compute_model.sample(stage_id, phase)
        end_time = engine.now + latency
        device.busy_until = end_time

        engine.schedule(Event(
            time=end_time,
            kind=EventKind.COMPUTE_END,
            payload={"stage_id": stage_id, "microbatch_id": mb_id,
                     "phase": phase, "compute_start": engine.now},
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

        # Mark completed
        key = (stage_id, mb_id, phase)
        self._state.in_flight.discard(key)
        self._state.completed.add(key)

        # Determine next action based on phase and position
        stage_idx = self.stage_order.index(stage_id)

        if phase == Phase.FORWARD:
            if stage_idx < len(self.stage_order) - 1:
                # Transfer activations to next stage
                next_stage_id = self.stage_order[stage_idx + 1]
                # Mark destination as in-flight so scheduler won't double-book
                self._state.in_flight.add((next_stage_id, mb_id, Phase.FORWARD))
                self._schedule_transfer(mb_id, phase, stage_id, next_stage_id)
            # Last stage forward done: scheduler decides when backward starts
        else:  # BACKWARD
            if stage_idx > 0:
                # Transfer gradients to previous stage
                prev_stage_id = self.stage_order[stage_idx - 1]
                # Mark destination as in-flight
                self._state.in_flight.add((prev_stage_id, mb_id, Phase.BACKWARD))
                self._schedule_transfer(mb_id, phase, stage_id, prev_stage_id)
            else:
                # Stage 0 backward done: micro-batch complete
                self._batch_done_count += 1
                if self._batch_done_count >= self._expected_done_count:
                    self.collector.record_batch_completion(engine.now)

        # Ask scheduler for more work
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

        link = self.topology.link(src, dst)
        transfer_time = link.sample_transfer_time(self.activation_size_mb)

        engine.schedule(Event(
            time=engine.now + transfer_time,
            kind=EventKind.TRANSFER_END,
            payload={**event.payload, "transfer_start": engine.now},
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

        # Start computation on destination stage (already marked in_flight)
        engine.schedule(Event(
            time=engine.now,
            kind=EventKind.COMPUTE_START,
            payload={"stage_id": to_stage, "microbatch_id": mb_id,
                     "phase": phase},
        ))
