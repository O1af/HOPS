"""Timing and resource availability helpers for pipeline execution."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from hops.core.types import EventKind, Phase
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel


@dataclass
class InFlightTransfer:
    transfer_id: int
    link_key: tuple[str, str]
    size_mb: float
    remaining_mb: float
    last_recalc_time: float
    generation: int
    event_payload: dict = field(default_factory=dict)
    reschedule_kind: EventKind = EventKind.TRANSFER_END


class TimingModel:
    """Compute/transfer timing with failure-aware delays and bandwidth contention.

    Compute and transfer resources are independent — a device can compute
    while simultaneously sending/receiving data (overlap).  Bandwidth
    contention is modelled per-link: concurrent transfers share the link's
    bandwidth equally, and are re-evaluated when contention changes.
    """

    def __init__(self, topology: Topology, compute_model: ComputeModel,
                 rng: np.random.Generator):
        self.topology = topology
        self.compute_model = compute_model
        self.rng = rng
        self.failure_engine = None
        self._next_transfer_id: int = 0
        self._in_flight: dict[int, InFlightTransfer] = {}
        self._link_transfers: dict[tuple[str, str], set[int]] = defaultdict(set)

    def set_failure_engine(self, failure_engine) -> None:
        self.failure_engine = failure_engine

    def reserve_compute(self, *, now: float, stage_id: int, device_id: str,
                        phase: Phase) -> tuple[float, float]:
        device = self.topology.device(device_id)
        start_time = max(now, device.busy_until)
        if self.failure_engine is not None:
            recovery_time = self.failure_engine.next_device_recovery_time(device_id)
            if recovery_time is not None and recovery_time > start_time:
                start_time = recovery_time

        if start_time > now:
            return start_time, start_time

        duration = self.compute_model.sample(stage_id, phase, self.rng)
        end_time = start_time + duration
        device.busy_until = end_time
        return start_time, end_time

    def reserve_device(self, *, now: float, device_id: str,
                       duration: float) -> tuple[float, float]:
        """Reserve a device for a fixed duration, respecting failures."""
        device = self.topology.device(device_id)
        start_time = max(now, device.busy_until)
        if self.failure_engine is not None:
            recovery_time = self.failure_engine.next_device_recovery_time(device_id)
            if recovery_time is not None and recovery_time > start_time:
                start_time = recovery_time

        if start_time > now:
            return start_time, start_time

        end_time = start_time + duration
        device.busy_until = end_time
        return start_time, end_time

    def reserve_transfer(self, *, now: float, src_device: str, dst_device: str,
                         size_mb: float) -> tuple[float, float, int]:
        """Reserve a link for a transfer with contention-aware bandwidth.

        Transfers do NOT block device compute (overlap is allowed).
        Returns (start_time, end_time, transfer_id).
        """
        start_time = now
        if self.failure_engine is not None:
            recovery_time = self.failure_engine.next_link_recovery_time(src_device, dst_device)
            if recovery_time is not None and recovery_time > start_time:
                start_time = recovery_time

        if start_time > now:
            return start_time, start_time, -1

        link = self.topology.link(src_device, dst_device)
        link.active_transfers += 1
        effective_bw = link.bandwidth_gbps / link.active_transfers
        base_ms = link.base_latency_us / 1000.0
        transfer_ms = (size_mb * 8.0) / effective_bw
        duration = base_ms + transfer_ms + link.jitter.sample(self.rng)
        duration *= self.topology.transfer_penalty(src_device, dst_device).transfer_scale
        end_time = start_time + duration

        tid = self._next_transfer_id
        self._next_transfer_id += 1
        link_key = (src_device, dst_device)
        self._in_flight[tid] = InFlightTransfer(
            transfer_id=tid,
            link_key=link_key,
            size_mb=size_mb,
            remaining_mb=size_mb,
            last_recalc_time=start_time,
            generation=0,
        )
        self._link_transfers[link_key].add(tid)
        return start_time, end_time, tid

    def set_transfer_payload(self, transfer_id: int, payload: dict,
                             kind: EventKind) -> None:
        t = self._in_flight.get(transfer_id)
        if t is not None:
            t.event_payload = payload
            t.reschedule_kind = kind

    def is_transfer_current(self, transfer_id: int, generation: int) -> bool:
        t = self._in_flight.get(transfer_id)
        if t is None:
            return True
        return t.generation == generation

    def release_transfer(self, src_device: str, dst_device: str,
                         transfer_id: int = -1,
                         now: float = 0.0) -> list[tuple[int, float, int]]:
        """Release a link after transfer completes. Returns reschedule list.

        Each entry is (transfer_id, new_end_time, new_generation) for
        in-flight transfers that should complete sooner due to reduced
        contention.
        """
        link = self.topology.link(src_device, dst_device)
        link_key = (src_device, dst_device)

        old_active = link.active_transfers
        link.active_transfers = max(0, link.active_transfers - 1)

        if transfer_id >= 0:
            self._in_flight.pop(transfer_id, None)
            self._link_transfers[link_key].discard(transfer_id)

        if transfer_id < 0 or link.active_transfers == 0 or old_active <= 1:
            return []

        penalty = self.topology.transfer_penalty(src_device, dst_device).transfer_scale
        old_bw = link.bandwidth_gbps / old_active
        new_bw = link.bandwidth_gbps / link.active_transfers

        reschedules: list[tuple[int, float, int]] = []
        for tid in list(self._link_transfers[link_key]):
            t = self._in_flight.get(tid)
            if t is None:
                continue
            elapsed = now - t.last_recalc_time
            if elapsed > 0 and penalty > 0:
                transferred_mb = (old_bw * elapsed) / (8.0 * penalty)
                t.remaining_mb = max(0.0, t.remaining_mb - transferred_mb)
            t.last_recalc_time = now

            if t.remaining_mb <= 1e-12:
                new_end = now
            else:
                new_transfer_ms = (t.remaining_mb * 8.0) / new_bw
                new_end = now + new_transfer_ms * penalty

            t.generation += 1
            reschedules.append((tid, new_end, t.generation))

        return reschedules
