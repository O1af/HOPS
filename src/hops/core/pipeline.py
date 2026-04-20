"""Pipeline model: stages, micro-batches, and forward/backward dataflow."""

from dataclasses import dataclass

import numpy as np

from hops.core.event_engine import EventEngine
from hops.core.scheduler import PipelineState, Scheduler
from hops.core.timing import TimingModel
from hops.core.types import AllreduceAlgo, Event, EventKind, Phase, Precision
from hops.latency.distributions import Distribution
from hops.metrics.collector import MetricsCollector


@dataclass
class Stage:
    id: int
    device_id: str


@dataclass
class PipelineBatchState:
    done_count: int = 0
    expected_done_count: int = 0
    optimizer_pending: int = 0
    allreduce_pending: int = 0
    in_optimizer_phase: bool = False
    accumulation_count: int = 0
    is_optimizer_batch: bool = False
    w_tasks_remaining: int = 0
    microbatch_offset: int = 0
    ring_chunk_size: float = 0.0
    ring_total_rounds: int = 0
    ring_current_round: int = 0


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
                 precision: Precision = Precision.FP32,
                 allreduce_algo: AllreduceAlgo = AllreduceAlgo.NAIVE,
                 iteration_barrier: Distribution | None = None):
        self._validate_stage_configuration(stages)
        self.stages = {s.id: s for s in stages}
        self.stage_order = [s.id for s in stages]
        self._stage_index = {sid: i for i, sid in enumerate(self.stage_order)}
        self.engine = engine
        self.topology = topology
        self._validate_topology_requirements()
        self.scheduler = scheduler
        self.collector = collector
        self.rng = rng if rng is not None else np.random.default_rng()
        self.timing_model = TimingModel(topology, compute_model, self.rng)

        # Precision scaling
        self.precision = precision
        self.activation_size_mb = activation_size_mb * precision.data_scale
        self.gradient_size_mb = gradient_size_mb * precision.data_scale

        # Optimizer config
        self.optimizer_latency = optimizer_latency
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.allreduce_algo = allreduce_algo
        # Host/framework stall applied before every batch except the first.
        self.iteration_barrier = iteration_barrier
        self._first_batch_started = False

        self._configure_scheduler(stages)
        self._use_w_split = scheduler.uses_w_split

        # Pre-compute unique ordered devices + device→stage lookup
        seen: dict[str, int] = {}
        for sid in self.stage_order:
            did = self.stages[sid].device_id
            if did not in seen:
                seen[did] = sid
        self._unique_devices = list(seen.keys())
        self._device_to_first_stage = seen

        # Memory tracking
        self.stage_memory_mb = stage_memory_mb or {}
        for stage_id, mem in self.stage_memory_mb.items():
            device = self.topology.device(self.stages[stage_id].device_id)
            device.allocate(mem * precision.weight_memory_overhead)
            self.collector.record_peak_memory(device.id, device.peak_memory_mb)

        self._state = PipelineState(num_stages=len(stages), num_microbatches=0)
        self._batch = PipelineBatchState()
        self._total_mb_issued = 0

        # Register event handlers
        engine.on(EventKind.COMPUTE_START, self._on_compute_start)
        engine.on(EventKind.COMPUTE_END, self._on_compute_end)
        engine.on(EventKind.TRANSFER_START, self._on_transfer_start)
        engine.on(EventKind.TRANSFER_END, self._on_transfer_end)
        engine.on(EventKind.ALLREDUCE_START, self._on_allreduce_start)
        engine.on(EventKind.ALLREDUCE_END, self._on_allreduce_end)
        engine.on(EventKind.OPTIMIZER_START, self._on_optimizer_start)
        engine.on(EventKind.OPTIMIZER_END, self._on_optimizer_end)
        engine.on(EventKind.BATCH_START, self._on_batch_start)

    def _configure_scheduler(self, stages: list[Stage]) -> None:
        rng = np.random.default_rng(0)
        num_stages = len(stages)

        def _mean_sample(stage_id: int, phase: Phase, n: int = 16) -> float:
            total = 0.0
            for _ in range(n):
                total += self.timing_model.compute_model.sample(stage_id, phase, rng)
            return total / n

        fwd_ms = [_mean_sample(s.id, Phase.FORWARD) for s in stages]
        b_ms = [_mean_sample(s.id, Phase.BACKWARD_B) for s in stages]
        w_ms = [_mean_sample(s.id, Phase.BACKWARD_W) for s in stages]
        bwd_ms = [b + w for b, w in zip(b_ms, w_ms)]

        device_ids = [s.device_id for s in stages]
        unique_devs: list[str] = []
        for d in device_ids:
            if d not in unique_devs:
                unique_devs.append(d)
        stage_device_idx = [unique_devs.index(d) for d in device_ids]

        try:
            self.scheduler.configure({
                "num_stages": num_stages,
                "num_microbatches": 0,
                "fwd_ms": fwd_ms,
                "bwd_ms": bwd_ms,
                "b_ms": b_ms,
                "w_ms": w_ms,
                "stage_device_idx": stage_device_idx,
                "num_devices": len(unique_devs),
            })
        except NotImplementedError:
            pass

    @staticmethod
    def _validate_stage_configuration(stages: list[Stage]) -> None:
        if not stages:
            raise ValueError("Pipeline must contain at least one stage")

        stage_ids = [stage.id for stage in stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError(f"Duplicate stage IDs are not allowed: {stage_ids}")

        expected_ids = list(range(len(stages)))
        if stage_ids != expected_ids:
            raise ValueError(
                "Stage IDs must be contiguous zero-based integers in pipeline order. "
                f"Expected {expected_ids}, got {stage_ids}"
            )

    def _validate_topology_requirements(self) -> None:
        for stage in self.stages.values():
            try:
                self.topology.device(stage.device_id)
            except KeyError as exc:
                raise ValueError(
                    f"Stage {stage.id} references unknown device {stage.device_id!r}"
                ) from exc

        for from_stage, to_stage in zip(self.stage_order, self.stage_order[1:]):
            src = self.stages[from_stage].device_id
            dst = self.stages[to_stage].device_id
            for required_src, required_dst, phase_name in (
                (src, dst, "forward"),
                (dst, src, "backward"),
            ):
                try:
                    self.topology.link(required_src, required_dst)
                except KeyError as exc:
                    raise ValueError(
                        f"Pipeline {phase_name} transfer requires a link from "
                        f"{required_src!r} to {required_dst!r}"
                    ) from exc

    def start_batch(self, num_microbatches: int) -> None:
        """Begin a training step with N micro-batches."""
        self._state = PipelineState.initialize(
            num_stages=len(self.stages),
            num_microbatches=num_microbatches,
            use_w_split=self._use_w_split,
        )
        if hasattr(self.scheduler, "on_batch_start"):
            self.scheduler.on_batch_start(num_microbatches)
        self._batch.done_count = 0
        self._batch.expected_done_count = num_microbatches
        self._batch.microbatch_offset = self._total_mb_issued
        self._total_mb_issued += num_microbatches
        self._batch.optimizer_pending = 0
        self._batch.allreduce_pending = 0
        self._batch.in_optimizer_phase = False
        self._batch.w_tasks_remaining = (
            num_microbatches * len(self.stages) if self._use_w_split else 0
        )
        self._batch.ring_chunk_size = 0.0
        self._batch.ring_total_rounds = 0
        self._batch.ring_current_round = 0

        # Gradient accumulation
        self._batch.accumulation_count += 1
        self._batch.is_optimizer_batch = (
            self.optimizer_latency is not None
            and self._batch.accumulation_count >= self.gradient_accumulation_steps
        )

        barrier_delay = 0.0
        if self.iteration_barrier is not None and self._first_batch_started:
            barrier_delay = max(0.0, self.iteration_barrier.sample(self.rng))
        self._first_batch_started = True

        if barrier_delay > 0.0:
            self.engine.schedule(Event(
                time=self.engine.now + barrier_delay,
                kind=EventKind.BATCH_START,
                payload={},
            ))
        else:
            self._issue_ready_tasks()

    def _on_batch_start(self, event: Event, engine: EventEngine) -> None:
        self._issue_ready_tasks()

    @property
    def batch_complete(self) -> bool:
        if self._batch.expected_done_count <= 0:
            return False
        if self._batch.done_count < self._batch.expected_done_count:
            return False
        if self._use_w_split and self._batch.w_tasks_remaining > 0:
            return False
        if self._batch.is_optimizer_batch:
            return (
                self._batch.in_optimizer_phase
                and self._batch.optimizer_pending == 0
                and self._batch.allreduce_pending == 0
            )
        return True

    def set_failure_engine(self, failure_engine) -> None:
        self.timing_model.set_failure_engine(failure_engine)

    @property
    def _ring_chunk_size(self) -> float:
        return self._batch.ring_chunk_size

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
            self.topology.device(device_id).free(self.activation_size_mb)

        # Record metrics with globally unique MB ID
        global_mb_id = mb_id + self._batch.microbatch_offset
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

    def _complete_microbatch(self, global_microbatch_id: int, now: float) -> None:
        """Record microbatch completion and check if optimizer should start."""
        self._batch.done_count += 1
        self.collector.record_microbatch_completion(global_microbatch_id, now)
        self._check_optimizer_trigger()

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
                bwd_phase = Phase.BACKWARD_B if self._use_w_split else Phase.BACKWARD
                self._state.mark_ready(stage_id, microbatch_id, bwd_phase)

        elif phase in (Phase.BACKWARD, Phase.BACKWARD_B):
            if phase == Phase.BACKWARD_B:
                self._state.mark_ready(stage_id, microbatch_id, Phase.BACKWARD_W)
            if stage_idx > 0:
                prev_stage_id = self.stage_order[stage_idx - 1]
                self._schedule_transfer(microbatch_id, phase, stage_id, prev_stage_id)
            else:
                self._complete_microbatch(global_microbatch_id, now)

        elif phase == Phase.BACKWARD_W:
            self._batch.w_tasks_remaining -= 1
            self._check_optimizer_trigger()

        self._issue_ready_tasks()

    def _check_optimizer_trigger(self) -> None:
        """Start optimizer phase if all conditions are met."""
        if self._batch.done_count < self._batch.expected_done_count:
            return
        if self._use_w_split and self._batch.w_tasks_remaining > 0:
            return
        if not self._batch.is_optimizer_batch:
            return
        if self._batch.in_optimizer_phase:
            return
        self._batch.accumulation_count = 0
        self._start_optimizer_phase()

    def _validate_links(self, pairs: list[tuple[str, str]], context: str) -> None:
        """Ensure all required links exist in topology, or raise ValueError."""
        for src, dst in pairs:
            try:
                self.topology.link(src, dst)
            except KeyError:
                raise ValueError(
                    f"{context} requires a link from {src!r} to {dst!r}, "
                    f"but none exists in the topology"
                )

    def _start_optimizer_phase(self) -> None:
        """Kick off all-reduce followed by optimizer compute on each device."""
        self._batch.in_optimizer_phase = True

        if self.gradient_size_mb > 0 and len(self._unique_devices) > 1:
            if self.allreduce_algo == AllreduceAlgo.RING:
                self._start_ring_allreduce()
            else:
                self._start_naive_allreduce()
        else:
            self._start_optimizer_compute()

    def _start_naive_allreduce(self) -> None:
        """Adjacent pairwise all-reduce (original algorithm)."""
        pairs = []
        for i in range(len(self._unique_devices) - 1):
            pairs.append((self._unique_devices[i], self._unique_devices[i + 1]))
            pairs.append((self._unique_devices[i + 1], self._unique_devices[i]))
        self._validate_links(pairs, "Naive all-reduce")
        self._batch.allreduce_pending = len(pairs)
        for src, dst in pairs:
            self.engine.schedule(Event(
                time=self.engine.now,
                kind=EventKind.ALLREDUCE_START,
                payload={"src_device": src, "dst_device": dst,
                         "size_mb": self.gradient_size_mb},
            ))

    def _start_ring_allreduce(self) -> None:
        """Ring all-reduce: 2*(N-1) rounds, N parallel transfers per round."""
        N = len(self._unique_devices)
        ring_pairs = [(self._unique_devices[i], self._unique_devices[(i + 1) % N])
                      for i in range(N)]
        self._validate_links(ring_pairs, "Ring all-reduce")

        self._batch.ring_chunk_size = self.gradient_size_mb / N
        self._batch.ring_total_rounds = 2 * (N - 1)
        self._batch.ring_current_round = 0
        self._start_ring_round()

    def _start_ring_round(self) -> None:
        """Launch one round of ring all-reduce (N parallel transfers)."""
        N = len(self._unique_devices)
        self._batch.allreduce_pending = N
        for i in range(N):
            src = self._unique_devices[i]
            dst = self._unique_devices[(i + 1) % N]
            self.engine.schedule(Event(
                time=self.engine.now,
                kind=EventKind.ALLREDUCE_START,
                payload={"src_device": src, "dst_device": dst,
                         "size_mb": self._batch.ring_chunk_size},
            ))

    def _on_allreduce_start(self, event: Event, engine: EventEngine) -> None:
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]
        size_mb = event.payload["size_mb"]
        start_time, end_time, tid = self.timing_model.reserve_transfer(
            now=engine.now, src_device=src, dst_device=dst, size_mb=size_mb,
        )
        if start_time > engine.now:
            engine.schedule(Event(time=start_time, kind=EventKind.ALLREDUCE_START,
                                  payload=event.payload))
            return

        base_payload = {**event.payload, "transfer_start": start_time,
                        "transfer_id": tid}
        self.timing_model.set_transfer_payload(tid, base_payload,
                                               EventKind.ALLREDUCE_END)
        engine.schedule(Event(
            time=end_time,
            kind=EventKind.ALLREDUCE_END,
            payload={**base_payload, "generation": 0},
        ))

    def _on_allreduce_end(self, event: Event, engine: EventEngine) -> None:
        tid = event.payload.get("transfer_id", -1)
        generation = event.payload.get("generation", 0)

        if tid >= 0 and not self.timing_model.is_transfer_current(tid, generation):
            return

        transfer_start = event.payload["transfer_start"]
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]

        reschedules = self.timing_model.release_transfer(
            src, dst, tid, engine.now)
        self._reschedule_link_transfers(reschedules, engine)

        self.collector.record_transfer(
            None, Phase.OPTIMIZER, src, dst, transfer_start, engine.now)

        self._batch.allreduce_pending -= 1
        if self._batch.allreduce_pending == 0:
            if self.allreduce_algo == AllreduceAlgo.RING:
                self._batch.ring_current_round += 1
                if self._batch.ring_current_round < self._batch.ring_total_rounds:
                    self._start_ring_round()
                    return
            self._start_optimizer_compute()

    def _start_optimizer_compute(self) -> None:
        """Schedule optimizer weight update on each device."""
        self._batch.optimizer_pending = len(self._unique_devices)
        for device_id in self._unique_devices:
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
        stage_id = self._device_to_first_stage[device_id]
        self.collector.record_compute(stage_id, None, Phase.OPTIMIZER, device_id,
                                      start_time, engine.now)
        self._batch.optimizer_pending -= 1

    def _reschedule_link_transfers(self, reschedules: list, engine: EventEngine) -> None:
        for tid, new_end, new_gen in reschedules:
            t = self.timing_model._in_flight.get(tid)
            if t is not None and t.event_payload:
                engine.schedule(Event(
                    time=new_end,
                    kind=t.reschedule_kind,
                    payload={**t.event_payload, "generation": new_gen},
                ))

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
        start_time, end_time, tid = self.timing_model.reserve_transfer(
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

        base_payload = {**event.payload, "transfer_start": start_time,
                        "transfer_id": tid}
        self.timing_model.set_transfer_payload(tid, base_payload,
                                               EventKind.TRANSFER_END)
        engine.schedule(Event(
            time=end_time,
            kind=EventKind.TRANSFER_END,
            payload={**base_payload, "generation": 0},
        ))

    def _on_transfer_end(self, event: Event, engine: EventEngine) -> None:
        tid = event.payload.get("transfer_id", -1)
        generation = event.payload.get("generation", 0)

        if tid >= 0 and not self.timing_model.is_transfer_current(tid, generation):
            return

        mb_id = event.payload["microbatch_id"]
        phase = event.payload["phase"]
        to_stage = event.payload["to_stage"]
        transfer_start = event.payload["transfer_start"]
        src = event.payload["src_device"]
        dst = event.payload["dst_device"]

        reschedules = self.timing_model.release_transfer(
            src, dst, tid, engine.now)
        self._reschedule_link_transfers(reschedules, engine)

        global_mb_id = mb_id + self._batch.microbatch_offset
        self.collector.record_transfer(
            global_mb_id, phase, src, dst, transfer_start, engine.now)

        # All transfer types mark the same phase ready at the destination
        self._state.mark_ready(to_stage, mb_id, phase)
        self._issue_ready_tasks()
