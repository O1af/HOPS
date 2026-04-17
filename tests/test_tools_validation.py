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


def test_parse_link_bench_ignores_mixed_torchrun_log_lines(tmp_path: Path) -> None:
    path = tmp_path / "link_bench" / "pair_0_1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join([
            "/home/ubuntu/megatron-env/lib/python3.12/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier()",
            "[rank0]:[W417 18:58:22.016887355 ProcessGroupNCCL.cpp:5188] Guessing device ID",
            json.dumps({
                "label": "pair_0_1",
                "mode": "p2p",
                "rank": 0,
                "world_size": 2,
                "hostname": "nodeA",
                "size_mb": 1,
                "dtype": "bfloat16",
                "local_rank": 0,
                "warmup": 0,
                "iters": 1,
                "mean_ms": 0.5,
                "p50_ms": 0.5,
                "p99_ms": 0.5,
                "min_ms": 0.5,
                "max_ms": 0.5,
                "std_ms": 0.0,
            }),
        ])
        + "\n",
        encoding="utf-8",
    )

    rows = parse_link_bench._iter_jsonl_rows([path])
    assert len(rows) == 1
    assert rows[0]["label"] == "pair_0_1"


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

    loaded = load_raw_megatron_events(trace_dir, strip_warmup=False)
    fits = derive_megatron_stats.fit_stage_distributions(loaded, phase=Phase.FORWARD)
    by_stage = {f.stage: f for f in fits}
    assert by_stage[0].mean_ms == pytest.approx(11.0)
    assert by_stage[1].mean_ms == pytest.approx(20.5)
    assert by_stage[0].count == 3
    assert by_stage[1].count == 2

    document = derive_megatron_stats.build_overlay(fits)
    assert document["pipeline"]["stages"][0]["compute"]["mode"] == "explicit"
    assert document["pipeline"]["stages"][0]["compute"]["distribution"]["type"] == "normal"
    assert "backward" not in document["pipeline"]["stages"][0]


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


def _compute_event(*, stage: int, iteration: int, microbatch: int, phase: str,
                   start_ns: int, dur_ms: float, device: str) -> dict:
    return {
        "rank": stage, "stage": stage, "iteration": iteration, "microbatch": microbatch,
        "event_type": "compute", "phase": phase,
        "start_wall_ns": start_ns,
        "end_wall_ns": start_ns + int(dur_ms * 1_000_000),
        "device_id": device,
    }


def test_derive_megatron_stats_emits_backward_overlay(tmp_path: Path) -> None:
    trace_dir = tmp_path / "megatron_trace"
    events: list[dict] = []
    base = 1_000_000_000
    iter_step = 100_000_000
    for it in range(2, 6):
        t = base + (it - 2) * iter_step
        events.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                     phase="FORWARD", start_ns=t,
                                     dur_ms=10.0, device="a10g_node0_gpu0"))
        events.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                     phase="BACKWARD", start_ns=t + 30_000_000,
                                     dur_ms=12.0, device="a10g_node0_gpu0"))
        events.append(_compute_event(stage=1, iteration=it, microbatch=0,
                                     phase="FORWARD", start_ns=t + 15_000_000,
                                     dur_ms=20.0, device="a10g_node1_gpu0"))
        events.append(_compute_event(stage=1, iteration=it, microbatch=0,
                                     phase="BACKWARD", start_ns=t + 50_000_000,
                                     dur_ms=18.0, device="a10g_node1_gpu0"))
    _write_jsonl(trace_dir / "rank0.jsonl", events)

    output = tmp_path / "stage_timings.yaml"
    forward, backward, optimizer = derive_megatron_stats.write_stage_timings_overlay(
        trace_dir, output
    )
    assert {fit.stage for fit in forward} == {0, 1}
    assert {fit.stage for fit in backward} == {0, 1}
    assert optimizer is None
    by_stage = {fit.stage: fit for fit in backward}
    assert by_stage[0].mean_ms == pytest.approx(12.0)
    assert by_stage[1].mean_ms == pytest.approx(18.0)

    overlay = yaml.safe_load(output.read_text())
    stage0 = overlay["pipeline"]["stages"][0]
    assert stage0["compute"]["distribution"]["mean"] == pytest.approx(10.0)
    assert stage0["backward"]["distribution"]["mean"] == pytest.approx(12.0)


def test_warmup_strip_drops_leading_outliers(tmp_path: Path) -> None:
    from hops.megatron.importer import (
        load_raw_megatron_events,
        strip_warmup_iterations,
    )

    trace_dir = tmp_path / "megatron_trace"
    events: list[dict] = []
    # iter 0 is 5x slower than the rest -> warmup
    base = 1_000_000_000
    iter_dur = {0: 50.0, 1: 10.0, 2: 11.0, 3: 9.0, 4: 10.5}
    for it, dur in iter_dur.items():
        events.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                     phase="FORWARD",
                                     start_ns=base + it * 200_000_000,
                                     dur_ms=dur, device="d0"))
    _write_jsonl(trace_dir / "rank0.jsonl", events)

    loaded = load_raw_megatron_events(trace_dir, strip_warmup=False)
    kept, dropped = strip_warmup_iterations(loaded)
    assert dropped == [0]
    assert {e.iteration for e in kept} == {1, 2, 3, 4}

    auto_stripped = load_raw_megatron_events(trace_dir)
    assert {e.iteration for e in auto_stripped} == {1, 2, 3, 4}


def test_warmup_strip_preserves_clean_run(tmp_path: Path) -> None:
    from hops.megatron.importer import strip_warmup_iterations, load_raw_megatron_events

    trace_dir = tmp_path / "megatron_trace"
    events: list[dict] = []
    base = 1_000_000_000
    for it, dur in enumerate([10.0, 11.0, 9.5, 10.5, 10.0]):
        events.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                     phase="FORWARD",
                                     start_ns=base + it * 200_000_000,
                                     dur_ms=dur, device="d0"))
    _write_jsonl(trace_dir / "rank0.jsonl", events)

    loaded = load_raw_megatron_events(trace_dir, strip_warmup=False)
    kept, dropped = strip_warmup_iterations(loaded)
    assert dropped == []
    assert len(kept) == len(loaded)


def _compute_model_from_raw(raw: dict):
    from hops.config import parse_config
    from hops.runtime import build_runtime

    config = parse_config(raw)
    return build_runtime(config).pipeline.timing_model.compute_model


def test_explicit_backward_overrides_backward_factor() -> None:
    import numpy as np

    from hops.core.types import Phase

    raw = _base_config_dict()
    raw["pipeline"]["precision"] = "fp32"
    for stage_raw in raw["pipeline"]["stages"]:
        stage_raw["compute"] = {"mode": "explicit",
                                 "distribution": {"type": "constant", "value": 10.0}}
        stage_raw["backward"] = {"distribution": {"type": "constant", "value": 5.0}}

    model = _compute_model_from_raw(raw)
    rng = np.random.default_rng(0)
    # Without an explicit backward block, backward_factor=2.0 would yield 20.0;
    # the per-stage backward distribution should win and yield 5.0.
    assert model.sample(0, Phase.FORWARD, rng) == pytest.approx(10.0)
    assert model.sample(0, Phase.BACKWARD, rng) == pytest.approx(5.0)


def test_compute_model_falls_back_to_backward_factor_without_overlay() -> None:
    import numpy as np

    from hops.core.types import Phase

    raw = _base_config_dict()
    raw["pipeline"]["precision"] = "fp32"
    for stage_raw in raw["pipeline"]["stages"]:
        stage_raw["compute"] = {"mode": "explicit",
                                 "distribution": {"type": "constant", "value": 10.0}}

    model = _compute_model_from_raw(raw)
    rng = np.random.default_rng(0)
    assert model.sample(0, Phase.BACKWARD, rng) == pytest.approx(20.0)


def test_fit_optimizer_distribution_pools_across_ranks(tmp_path: Path) -> None:
    from hops.core.types import Phase
    from hops.megatron.importer import load_raw_megatron_events

    trace_dir = tmp_path / "megatron_trace"
    events_rank0: list[dict] = []
    events_rank1: list[dict] = []
    base = 1_000_000_000
    # Seed a forward event so the importer's warmup-strip has trailing data.
    for it in range(2, 6):
        t = base + (it - 2) * 200_000_000
        events_rank0.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                           phase="FORWARD", start_ns=t,
                                           dur_ms=10.0, device="d0"))
        events_rank1.append(_compute_event(stage=1, iteration=it, microbatch=0,
                                           phase="FORWARD", start_ns=t,
                                           dur_ms=10.0, device="d1"))
        events_rank0.append(_compute_event(stage=0, iteration=it, microbatch=None,
                                           phase="OPTIMIZER",
                                           start_ns=t + 100_000_000,
                                           dur_ms=4.0, device="d0"))
        events_rank1.append(_compute_event(stage=1, iteration=it, microbatch=None,
                                           phase="OPTIMIZER",
                                           start_ns=t + 100_000_000,
                                           dur_ms=6.0, device="d1"))
    _write_jsonl(trace_dir / "rank0.jsonl", events_rank0)
    _write_jsonl(trace_dir / "rank1.jsonl", events_rank1)

    loaded = load_raw_megatron_events(trace_dir, strip_warmup=False)
    fit = derive_megatron_stats.fit_optimizer_distribution(loaded, phase=Phase.OPTIMIZER)
    assert fit is not None
    assert fit.count == 8
    assert fit.mean_ms == pytest.approx(5.0)


def test_write_stage_timings_overlay_emits_optimizer_overlay(tmp_path: Path) -> None:
    trace_dir = tmp_path / "megatron_trace"
    events: list[dict] = []
    base = 1_000_000_000
    for it in range(2, 6):
        t = base + (it - 2) * 200_000_000
        events.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                     phase="FORWARD", start_ns=t,
                                     dur_ms=10.0, device="a10g_node0_gpu0"))
        events.append(_compute_event(stage=1, iteration=it, microbatch=0,
                                     phase="FORWARD", start_ns=t + 15_000_000,
                                     dur_ms=20.0, device="a10g_node1_gpu0"))
        events.append(_compute_event(stage=0, iteration=it, microbatch=None,
                                     phase="OPTIMIZER",
                                     start_ns=t + 100_000_000,
                                     dur_ms=4.0, device="a10g_node0_gpu0"))
        events.append(_compute_event(stage=1, iteration=it, microbatch=None,
                                     phase="OPTIMIZER",
                                     start_ns=t + 100_000_000,
                                     dur_ms=6.0, device="a10g_node1_gpu0"))
    _write_jsonl(trace_dir / "rank0.jsonl", events)

    stage_output = tmp_path / "stage_timings.yaml"
    optimizer_output = tmp_path / "optimizer.yaml"
    forward, _backward, optimizer_fit = derive_megatron_stats.write_stage_timings_overlay(
        trace_dir, stage_output, 0, optimizer_output
    )
    assert forward
    assert optimizer_fit is not None
    assert optimizer_fit.mean_ms == pytest.approx(5.0)

    overlay = yaml.safe_load(optimizer_output.read_text())
    assert overlay["optimizer"]["enabled"] is True
    assert overlay["optimizer"]["update"]["type"] == "normal"
    assert overlay["optimizer"]["update"]["mean"] == pytest.approx(5.0)


def test_optimizer_overlay_merges_and_parses(tmp_path: Path) -> None:
    base = _base_config_dict()
    base_path = tmp_path / "hops.base.yaml"
    base_path.write_text(yaml.safe_dump(base))

    optimizer_overlay = tmp_path / "optimizer.yaml"
    optimizer_overlay.write_text(yaml.safe_dump({
        "optimizer": {
            "enabled": True,
            "update": {"type": "normal", "mean": 5.0, "std": 0.5},
        }
    }))

    output_path = tmp_path / "merged.yaml"
    materialize_hops_variant.materialize(base_path, [optimizer_overlay], output_path)

    reloaded = yaml.safe_load(output_path.read_text())
    config = parse_config(reloaded)
    assert config.optimizer.enabled is True
    assert config.optimizer.update_distribution["mean"] == pytest.approx(5.0)


def test_write_stage_timings_overlay_skips_optimizer_when_absent(tmp_path: Path) -> None:
    trace_dir = tmp_path / "megatron_trace"
    events: list[dict] = []
    base = 1_000_000_000
    for it in range(2, 6):
        t = base + (it - 2) * 200_000_000
        events.append(_compute_event(stage=0, iteration=it, microbatch=0,
                                     phase="FORWARD", start_ns=t,
                                     dur_ms=10.0, device="d0"))
    _write_jsonl(trace_dir / "rank0.jsonl", events)

    stage_output = tmp_path / "stage_timings.yaml"
    optimizer_output = tmp_path / "optimizer.yaml"
    _forward, _backward, optimizer_fit = derive_megatron_stats.write_stage_timings_overlay(
        trace_dir, stage_output, 0, optimizer_output
    )
    assert optimizer_fit is None
    assert not optimizer_output.exists()
