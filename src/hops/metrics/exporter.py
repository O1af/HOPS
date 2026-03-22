"""Trace export helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from hops.metrics.collector import MetricsCollector


class TraceExporter:
    """Serialize raw collector events to CSV."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def write_csv(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "event_type",
                    "phase",
                    "stage_id",
                    "microbatch_id",
                    "device_id",
                    "src_device",
                    "dst_device",
                    "start_time_ms",
                    "end_time_ms",
                    "duration_ms",
                    "target_id",
                    "recovery_time_ms",
                ],
            )
            writer.writeheader()
            for record in self.collector.computes:
                writer.writerow({
                    "event_type": "compute",
                    "phase": record.phase.name,
                    "stage_id": record.stage_id,
                    "microbatch_id": record.microbatch_id,
                    "device_id": record.device_id,
                    "src_device": "",
                    "dst_device": "",
                    "start_time_ms": record.start_time,
                    "end_time_ms": record.end_time,
                    "duration_ms": record.end_time - record.start_time,
                    "target_id": "",
                    "recovery_time_ms": "",
                })
            for record in self.collector.transfers:
                writer.writerow({
                    "event_type": "transfer",
                    "phase": record.phase.name,
                    "stage_id": "",
                    "microbatch_id": record.microbatch_id,
                    "device_id": "",
                    "src_device": record.src_device,
                    "dst_device": record.dst_device,
                    "start_time_ms": record.start_time,
                    "end_time_ms": record.end_time,
                    "duration_ms": record.end_time - record.start_time,
                    "target_id": "",
                    "recovery_time_ms": "",
                })
            for record in self.collector.failures:
                writer.writerow({
                    "event_type": "failure",
                    "phase": "",
                    "stage_id": "",
                    "microbatch_id": "",
                    "device_id": "",
                    "src_device": "",
                    "dst_device": "",
                    "start_time_ms": record.time,
                    "end_time_ms": record.time + record.recovery_time,
                    "duration_ms": record.recovery_time,
                    "target_id": record.target_id,
                    "recovery_time_ms": record.recovery_time,
                })
