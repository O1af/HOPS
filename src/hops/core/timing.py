"""Timing and resource availability helpers for pipeline execution."""

import numpy as np

from hops.core.types import Phase
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel


class TimingModel:
    """Current sequential compute/transfer timing with failure-aware delays."""

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
        start_time = now
        if self.failure_engine is not None:
            recovery_time = self.failure_engine.next_link_recovery_time(src_device, dst_device)
            if recovery_time is not None and recovery_time > start_time:
                start_time = recovery_time

        if start_time > now:
            return start_time, start_time

        link = self.topology.link(src_device, dst_device)
        end_time = start_time + link.sample_transfer_time(size_mb, self.rng)
        return start_time, end_time
