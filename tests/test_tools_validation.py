"""Tests for the experiments/tools validation toolchain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from hops.config import parse_config


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "experiments" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import compare_run  # noqa: E402
import derive_megatron_stats  # noqa: E402
import materialize_hops_variant  # noqa: E402
import parse_link_bench  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _base_config_dict() -> dict:
    return {
        "simulation": {"batches": 2, "microbatches": 2, "seed": 0},
        "pipeline": {
            "schedule": "1f1b",
            "precision": "bf16",
            "backward_factor": 2.0,
            "model": {"hidden_dim": 256, "seq_len": 256},
            "stages": [
                {
                    "device": "a10g_node0_gpu0",
                    "weights_mb": 64.0,
                    "compute": {"mode": "analytical", "tflop": 0.05, "memory_mb": 32.0},
                },
                {
                    "device": "a10g_node1_gpu0",
                    "weights_mb": 64.0,
                    "compute": {"mode": "analytical", "tflop": 0.05, "memory_mb": 32.0},
                },
            ],
        },
        "hardware": {
            "devices": [
                {"id": "a10g_node0_gpu0", "gpu": "a10g", "node": "a10g_node0", "socket": 0},
                {"id": "a10g_node1_gpu0", "gpu": "a10g", "node": "a10g_node1", "socket": 0},
            ],
            "interconnect": {"same_node": "pcie", "cross_node": "ethernet"},
        },
        "optimizer": {"enabled": False},
        "failure": {"enabled": False},
        "output": {},
    }


def test_parse_link_bench_derives_bidirectional_overrides(tmp_path: Path) -> None:
    input_dir = tmp_path / "link_bench"
    _write_jsonl(
        input_dir / "pair_0_1.jsonl",
        [
            {"label": "pair_0_1", "mode": "p2p", "rank": 0, "world_size": 2,
             "hostname": "nodeA", "size_mb": 1, "dtype": "bfloat16", "local_rank": 0,
             "warmup": 0, "iters": 1, "mean_ms": 0.5, "p50_ms": 0.5, "p99_ms": 0.5,
             "min_ms": 0.5, "max_ms": 0.5, "std_ms": 0.0},
            {"label": "pair_0_1", "mode": "p2p", "rank": 0, "world_size": 2,
             "hostname": "nodeA", "size_mb": 64, "dtype": "bfloat16", "local_rank": 0,
             "warmup": 0, "iters": 1, "mean_ms": 54.237, "p50_ms": 54.237, "p99_ms": 55.0,
             "min_ms": 54.0, "max_ms": 55.0, "std_ms": 0.2},
            {"label": "pair_0_1", "mode": "p2p", "rank": 1, "world_size": 2,
             "hostname": "nodeB", "size_mb": 1, "dtype": "bfloat16", "local_rank": 0,
             "warmup": 0, "iters": 1, "mean_ms": 0.5, "p50_ms": 0.5, "p99_ms": 0.5,
             "min_ms": 0.5, "max_ms": 0.5, "std_ms": 0.0},
            {"label": "pair_0_1", "mode": "p2p", "rank": 1, "world_size": 2,
             "hostname": "nodeB", "size_mb": 64, "dtype": "bfloat16", "local_rank": 0,
             "warmup": 0, "iters": 1, "mean_ms": 54.237, "p50_ms": 54.237, "p99_ms": 55.0,
             "min_ms": 54.0, "max_ms": 55.0, "std_ms": 0.2},
        ],
    )
    trace_map = {"stages": {"0": "a10g_node0_gpu0", "1": "a10g_node1_gpu0"}}
    (tmp_path / "trace_map.json").write_text(json.dumps(trace_map))

    rows = parse_link_bench._iter_jsonl_rows(sorted(input_dir.glob("*.jsonl")))
    measurements = parse_link_bench.derive_pair_measurements(rows)
    assert len(measurements) == 1
    m = measurements[0]
    assert (m.stage_src, m.stage_dst) == (0, 1)
    # 64 MiB * 8 / 54.237 ms ≈ 9.44 Gbps
    assert m.bandwidth_gbps == pytest.approx(9.44, abs=0.05)
    assert m.latency_us == pytest.approx(500.0)

    document = parse_link_bench.build_overrides_yaml(
        measurements, parse_link_bench._stage_to_device(trace_map)
    )
    links = document["overrides"]["links"]
    assert len(links) == 2
    srcs = {link["src"] for link in links}
    dsts = {link["dst"] for link in links}
    assert srcs == dsts == {"a10g_node0_gpu0", "a10g_node1_gpu0"}


def test_parse_link_bench_requires_both_ranks(tmp_path: Path) -> None:
    rows = [
        {"label": "pair_0_1", "mode": "p2p", "rank": 0, "size_mb": 1,
         "hostname": "a", "p50_ms": 1.0},
    ]
    with pytest.raises(ValueError, match="rank 0 and rank 1"):
        parse_link_bench.derive_pair_measurements(rows)


def test_materialize_applies_links_and_stage_overlays(tmp_path: Path) -> None:
    base_path = tmp_path / "hops.base.yaml"
    base_path.write_text(yaml.safe_dump(_base_config_dict()))

    links_overlay = tmp_path / "links.yaml"
    links_overlay.write_text(yaml.safe_dump({
        "overrides": {
            "links": [
                {"src": "a10g_node0_gpu0", "dst": "a10g_node1_gpu0",
                 "bandwidth_gbps": 9.44, "latency_us": 500.0},
                {"src": "a10g_node1_gpu0", "dst": "a10g_node0_gpu0",
                 "bandwidth_gbps": 9.44, "latency_us": 500.0},
            ],
        },
    }))

    stage_overlay = tmp_path / "stage_timings.yaml"
    stage_overlay.write_text(yaml.safe_dump({
        "pipeline": {
            "stages": [
                {"id": 0, "compute": {"mode": "explicit",
                                       "distribution": {"type": "normal", "mean": 12.0, "std": 0.5}}},
                {"id": 1, "compute": {"mode": "explicit",
                                       "distribution": {"type": "normal", "mean": 13.5, "std": 0.6}}},
            ],
        },
    }))

    output_path = tmp_path / "merged.yaml"
    merged = materialize_hops_variant.materialize(
        base_path, [links_overlay, stage_overlay], output_path
    )

    reloaded = yaml.safe_load(output_path.read_text())
    parse_config(reloaded)

    assert merged["overrides"]["links"][0]["bandwidth_gbps"] == pytest.approx(9.44)
    assert reloaded["pipeline"]["stages"][0]["compute"]["mode"] == "explicit"
    assert reloaded["pipeline"]["stages"][0]["compute"]["distribution"]["mean"] == pytest.approx(12.0)
    assert reloaded["pipeline"]["stages"][0]["device"] == "a10g_node0_gpu0"
    assert reloaded["pipeline"]["stages"][0]["weights_mb"] == 64.0
    assert output_path.read_text().startswith("# GENERATED")


def test_materialize_fails_on_invalid_merged_config(tmp_path: Path) -> None:
    base = _base_config_dict()
    base_path = tmp_path / "hops.base.yaml"
    base_path.write_text(yaml.safe_dump(base))

    bad_overlay = tmp_path / "bad.yaml"
    bad_overlay.write_text(yaml.safe_dump({
        "pipeline": {
            "stages": [
                {"id": 0, "compute": {"mode": "explicit"}},  # missing distribution
            ],
        },
    }))

    with pytest.raises(ValueError):
        materialize_hops_variant.materialize(base_path, [bad_overlay], tmp_path / "out.yaml")


def test_derive_megatron_stats_fits_forward_per_stage(tmp_path: Path) -> None:
    trace_dir = tmp_path / "megatron_trace"
    events = []
    stage0_durations = [10, 12, 11]
    stage1_durations = [20, 21]
    t0 = 1_000_000_000
    for i, dur in enumerate(stage0_durations):
        events.append({
            "rank": 0, "stage": 0, "iteration": 3 + i, "microbatch": 0,
            "event_type": "compute", "phase": "FORWARD",
            "start_wall_ns": t0 + i * 100_000_000,
            "end_wall_ns": t0 + i * 100_000_000 + dur * 1_000_000,
            "device_id": "a10g_node0_gpu0",
        })
    for i, dur in enumerate(stage1_durations):
        events.append({
            "rank": 1, "stage": 1, "iteration": 3 + i, "microbatch": 0,
            "event_type": "compute", "phase": "FORWARD",
            "start_wall_ns": t0 + i * 100_000_000 + 5_000_000,
            "end_wall_ns": t0 + i * 100_000_000 + 5_000_000 + dur * 1_000_000,
            "device_id": "a10g_node1_gpu0",
        })
    events.append({
        "rank": 0, "stage": 0, "iteration": 3, "microbatch": 0,
        "event_type": "compute", "phase": "BACKWARD",
        "start_wall_ns": t0 + 400_000_000,
        "end_wall_ns": t0 + 400_000_000 + 50_000_000,
        "device_id": "a10g_node0_gpu0",
    })
    _write_jsonl(trace_dir / "rank0.jsonl", events)

    from hops.core.types import Phase
    from hops.megatron.importer import load_raw_megatron_events

    loaded = load_raw_megatron_events(trace_dir)
    fits = derive_megatron_stats.fit_stage_distributions(loaded, forward_phase=Phase.FORWARD)
    by_stage = {f.stage: f for f in fits}
    assert by_stage[0].mean_ms == pytest.approx(11.0)
    assert by_stage[1].mean_ms == pytest.approx(20.5)
    assert by_stage[0].count == 3
    assert by_stage[1].count == 2

    document = derive_megatron_stats.build_overlay(fits)
    assert document["pipeline"]["stages"][0]["compute"]["mode"] == "explicit"
    assert document["pipeline"]["stages"][0]["compute"]["distribution"]["type"] == "normal"


def _fake_summary(per_s: float, mean_ms: float, bubble: float = 0.1) -> dict:
    return {
        "completed_microbatches": 4,
        "throughput": {"per_ms": per_s / 1000.0, "per_s": per_s},
        "latency_ms": {"p50_ms": mean_ms, "p99_ms": mean_ms * 1.1, "mean_ms": mean_ms},
        "bubble_ratio": bubble,
        "time_ms": {"trace_duration_ms": 100.0, "makespan_ms": 100.0,
                     "compute_ms": 60.0, "transfer_ms": 20.0,
                     "communication_overhead_ratio": 0.2},
        "utilization": {"per_stage": {"0": 0.8, "1": 0.75}, "per_device": {}, "per_link": {}},
        "optimizer": {"allreduce_time_ms": 0.0, "weight_update_time_ms": 0.0},
        "failures": {"count": 0, "total_downtime_ms": 0.0, "lost_work_ms": None},
        "memory": {"peak_per_device_mb": {}},
        "contention": {"global_peak_concurrency": 0.0,
                         "global_contended_transfer_fraction": 0.0, "per_link": {}},
        "peak_in_flight_per_stage": {},
    }


def test_compare_run_builds_document_and_report(tmp_path: Path) -> None:
    job_dir = tmp_path / "output" / "42"
    derived = job_dir / "derived"
    derived.mkdir(parents=True)

    (job_dir / "megatron_summary.json").write_text(
        json.dumps(_fake_summary(per_s=25.0, mean_ms=40.0))
    )
    (derived / "hops_no_lookahead_summary.json").write_text(
        json.dumps(_fake_summary(per_s=28.0, mean_ms=36.0))
    )
    (derived / "hops_link_calibrated_summary.json").write_text(
        json.dumps(_fake_summary(per_s=26.0, mean_ms=39.0))
    )
    (derived / "hops_trace_replay_summary.json").write_text(
        json.dumps(_fake_summary(per_s=25.1, mean_ms=40.1))
    )

    comparison_path, report_path = compare_run.run(job_dir, links_source="test")
    document = json.loads(comparison_path.read_text())
    assert document["megatron"]["throughput_per_s"] == pytest.approx(25.0)
    no_lookahead = document["variants"]["no_lookahead"]
    assert no_lookahead["throughput_per_s"] == pytest.approx(28.0)
    assert no_lookahead["throughput_error_pct"] == pytest.approx(12.0, abs=0.01)
    assert document["calibration_manifest"]["links_source"] == "test"

    report = report_path.read_text()
    assert "no_lookahead" in report
    assert "Overfit guard" in report


def test_compare_run_tolerates_missing_variants(tmp_path: Path) -> None:
    job_dir = tmp_path / "output" / "7"
    derived = job_dir / "derived"
    derived.mkdir(parents=True)
    (derived / "hops_no_lookahead_summary.json").write_text(
        json.dumps(_fake_summary(per_s=28.0, mean_ms=36.0))
    )

    comparison_path, _ = compare_run.run(job_dir)
    document = json.loads(comparison_path.read_text())
    assert document["megatron"] is None
    assert document["variants"]["link_calibrated"] == {"status": "missing"}
    assert document["variants"]["no_lookahead"]["throughput_per_s"] == pytest.approx(28.0)
    assert document["variants"]["no_lookahead"]["throughput_error_pct"] is None
