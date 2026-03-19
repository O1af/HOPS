"""Timing and resource availability helpers for pipeline execution."""

import numpy as np

from hops.core.types import Phase
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel


class TimingModel:
    """Compute/transfer timing with failure-aware delays and bandwidth contention.

    Compute and transfer resources are independent — a device can compute
    while simultaneously sending/receiving data (overlap).  Bandwidth
    contention is modelled per-link: concurrent transfers share the link's
    bandwidth equally (estimated at transfer start time).
    """

    def __init__(self, topology: Topology, compute_model: ComputeModel,
                 rng: np.random.Generator):
        self.topology = topology
        self.compute_model = compute_model
        self.rng = rng
        self.failure_engine = None

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
                         size_mb: float) -> tuple[float, float]:
        """Reserve a link for a transfer with contention-aware bandwidth.

        Transfers do NOT block device compute (overlap is allowed).
        """
        start_time = now
        if self.failure_engine is not None:
            recovery_time = self.failure_engine.next_link_recovery_time(src_device, dst_device)
            if recovery_time is not None and recovery_time > start_time:
                start_time = recovery_time

        if start_time > now:
            return start_time, start_time

        link = self.topology.link(src_device, dst_device)
        link.active_transfers += 1
        effective_bw = link.bandwidth_gbps / link.active_transfers
        base_ms = link.base_latency_us / 1000.0
        transfer_ms = (size_mb * 8.0) / effective_bw
        duration = base_ms + transfer_ms + link.jitter.sample(self.rng)
        duration *= self.topology.transfer_penalty(src_device, dst_device).transfer_scale
        end_time = start_time + duration
        return start_time, end_time

    def release_transfer(self, src_device: str, dst_device: str) -> None:
        """Decrement active transfer count on a link after transfer completes."""
        link = self.topology.link(src_device, dst_device)
        link.active_transfers = max(0, link.active_transfers - 1)
