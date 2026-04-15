"""Tests for Megatron raw-trace conversion into HOPS-compatible outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from hops.megatron.compare import build_comparison, convert_job_dir
from hops.megatron.importer import import_megatron_trace_dir, load_raw_megatron_events


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def _job_dir_with_trace(tmp_path: Path) -> Path:
    job_dir = tmp_path / "output" / "26"
    trace_dir = job_dir / "megatron_trace"
    trace_dir.mkdir(parents=True)

    _write_jsonl(
        trace_dir / "rank0.jsonl",
        [
            {
                "rank": 0,
                "stage": 0,
                "iteration": 3,
                "microbatch": 0,
                "event_type": "compute",
                "phase": "FORWARD",
                "start_wall_ns": 1_000_000_000,
                "end_wall_ns": 1_010_000_000,
                "duration_ms": 10.0,
                "device_id": "h100_node0_gpu0",
                "hostname": "node0",
            },
            {
                "rank": 0,
                "stage": 0,
                "iteration": 3,
                "microbatch": 0,
                "event_type": "transfer",
                "phase": "FORWARD",
                "start_wall_ns": 1_010_000_000,
                "end_wall_ns": 1_012_000_000,
                "duration_ms": 2.0,
                "device_id": "h100_node0_gpu0",
                "src_device": "h100_node0_gpu0",
                "dst_device": "h100_node1_gpu0",
                "hostname": "node0",
            },
            {
                "rank": 0,
                "stage": 0,
                "iteration": 3,
                "microbatch": 0,
                "event_type": "compute",
                "phase": "BACKWARD",
                "start_wall_ns": 1_026_000_000,
                "end_wall_ns": 1_038_000_000,
                "duration_ms": 12.0,
                "device_id": "h100_node0_gpu0",
                "hostname": "node0",
            },
        ],
    )
    _write_jsonl(
        trace_dir / "rank1.jsonl",
        [
            {
                "rank": 1,
                "stage": 1,
                "iteration": 3,
                "microbatch": 0,
                "event_type": "compute",
                "phase": "FORWARD",
                "start_wall_ns": 1_012_000_000,
                "end_wall_ns": 1_020_000_000,
                "duration_ms": 8.0,
                "device_id": "h100_node1_gpu0",
                "hostname": "node1",
            },
            {
                "rank": 1,
                "stage": 1,
                "iteration": 3,
                "microbatch": 0,
                "event_type": "transfer",
                "phase": "BACKWARD",
                "start_wall_ns": 1_020_000_000,
                "end_wall_ns": 1_026_000_000,
                "duration_ms": 6.0,
                "device_id": "h100_node1_gpu0",
                "src_device": "h100_node1_gpu0",
                "dst_device": "h100_node0_gpu0",
                "hostname": "node1",
            },
            {
                "rank": 1,
                "stage": 1,
                "iteration": 3,
                "microbatch": 0,
                "event_type": "compute",
                "phase": "BACKWARD",
                "start_wall_ns": 1_020_000_000,
                "end_wall_ns": 1_026_000_000,
                "duration_ms": 6.0,
                "device_id": "h100_node1_gpu0",
                "hostname": "node1",
            },
        ],
    )
    return job_dir


def test_load_raw_megatron_events_reads_rank_files(tmp_path):
    job_dir = _job_dir_with_trace(tmp_path)
    events = load_raw_megatron_events(job_dir / "megatron_trace")

    assert len(events) == 6
    assert events[0].event_type == "compute"
    assert events[0].phase.name == "FORWARD"
    assert events[-1].phase.name == "BACKWARD"


def test_import_megatron_trace_dir_populates_collector(tmp_path):
    job_dir = _job_dir_with_trace(tmp_path)
    collector = import_megatron_trace_dir(job_dir / "megatron_trace")

    assert len(collector.computes) == 4
    assert len(collector.transfers) == 2
    assert collector.completed_microbatches == 1
    assert collector.computes[0].start_time == 0.0
    assert collector.microbatch_completion_times[0] == 38.0


def test_convert_job_dir_writes_hops_compatible_outputs(tmp_path):
    job_dir = _job_dir_with_trace(tmp_path)
    hops_summary = {
        "throughput": {"per_s": 20.0},
        "latency_ms": {"mean_ms": 40.0, "p50_ms": 40.0, "p99_ms": 40.0},
        "bubble_ratio": 0.2,
        "time_ms": {"communication_overhead_ratio": 0.5},
        "utilization": {
            "per_stage": {"0": 0.7, "1": 0.8},
            "per_link": {
                "h100_node0_gpu0->h100_node1_gpu0": 0.1,
                "h100_node1_gpu0->h100_node0_gpu0": 0.1,
            },
        },
    }
    with (job_dir / "hops_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(hops_summary, handle)

    outputs = convert_job_dir(job_dir)

    assert outputs.megatron_trace_csv.exists()
    assert outputs.megatron_summary_json.exists()
    assert outputs.comparison_json is not None and outputs.comparison_json.exists()

    with outputs.megatron_trace_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        header = rows[0].keys()

    assert list(header) == [
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
    ]

    summary = json.loads(outputs.megatron_summary_json.read_text(encoding="utf-8"))
    assert sorted(summary) == [
        "bubble_ratio",
        "completed_microbatches",
        "contention",
        "failures",
        "latency_ms",
        "memory",
        "optimizer",
        "peak_in_flight_per_stage",
        "throughput",
        "time_ms",
        "utilization",
    ]
    assert summary["completed_microbatches"] == 1
    assert summary["latency_ms"]["mean_ms"] == 38.0

    comparison = json.loads(outputs.comparison_json.read_text(encoding="utf-8"))
    assert "throughput" in comparison
    assert "bubble_ratio" in comparison
    assert "utilization" in comparison


def test_build_comparison_reports_selected_metric_deltas():
    megatron_summary = {
        "throughput": {"per_s": 10.0},
        "latency_ms": {"mean_ms": 50.0, "p50_ms": 49.0, "p99_ms": 60.0},
        "bubble_ratio": 0.3,
        "time_ms": {"communication_overhead_ratio": 0.4},
        "utilization": {
            "per_stage": {"0": 0.7},
            "per_link": {"a->b": 0.5},
        },
    }
    hops_summary = {
        "throughput": {"per_s": 8.0},
        "latency_ms": {"mean_ms": 40.0, "p50_ms": 40.0, "p99_ms": 50.0},
        "bubble_ratio": 0.2,
        "time_ms": {"communication_overhead_ratio": 0.5},
        "utilization": {
            "per_stage": {"0": 0.6},
            "per_link": {"a->b": 0.25},
        },
    }

    comparison = build_comparison(megatron_summary, hops_summary)
    assert comparison["throughput"]["delta"] == 2.0
    assert comparison["latency_ms"]["mean_ms"]["delta"] == 10.0
    assert comparison["utilization"]["per_stage"]["0"]["delta"] == pytest.approx(0.1)
