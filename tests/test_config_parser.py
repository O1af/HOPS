"""Tests for the canonical preset-first config schema."""

import pytest

from hops.config import parse_config, validate_config
from hops.runtime import build_runtime

from .conftest import make_canonical_config


def test_parse_valid_explicit_stage_config():
    config = make_canonical_config()
    parsed = parse_config(config)

    assert parsed.pipeline.schedule == "gpipe"
    assert parsed.pipeline.stages[0].compute_mode == "explicit"
    assert parsed.pipeline.stages[0].explicit is not None


def test_parse_valid_analytical_stage_config():
    config = make_canonical_config()
    config["pipeline"]["stages"][0]["compute"] = {
        "mode": "analytical",
        "tflop": 4.0,
        "memory_mb": 128.0,
        "efficiency": {"compute": 0.8, "memory": 0.9},
        "jitter": {"type": "constant", "value": 0.0},
    }

    parsed = parse_config(config)

    assert parsed.pipeline.stages[0].compute_mode == "analytical"
    assert parsed.pipeline.stages[0].analytical is not None
    assert parsed.pipeline.stages[0].analytical.tflop == 4.0


@pytest.mark.parametrize(
    ("memory_placement", "expected_kind"),
    [
        ({"kind": "local"}, "local"),
        ({"kind": "socket", "node": "node0", "socket": 1}, "socket"),
        ({"kind": "device", "device": "gpu0"}, "device"),
    ],
)
def test_parse_memory_placement_modes(memory_placement, expected_kind):
    config = make_canonical_config()
    config["pipeline"]["stages"][0]["compute"] = {
        "mode": "analytical",
        "tflop": 4.0,
        "memory_mb": 64.0,
    }
    config["pipeline"]["stages"][0]["memory_placement"] = memory_placement

    parsed = parse_config(config)

    assert parsed.pipeline.stages[0].memory_placement.kind == expected_kind


def test_validate_config_rejects_missing_compute_mode_definition():
    config = make_canonical_config()
    config["pipeline"]["stages"][0]["compute"] = {"mode": "analytical"}

    with pytest.raises(ValueError, match="mode=analytical requires tflop"):
        validate_config(config)


def test_build_runtime_rejects_unknown_device_preset():
    config = make_canonical_config()
    config["hardware"]["devices"][0]["gpu"] = "mystery-gpu"

    with pytest.raises(ValueError, match="Unknown hardware preset"):
        build_runtime(parse_config(config))


def test_parse_rejects_duplicate_device_ids():
    config = make_canonical_config()
    config["hardware"]["devices"].append(
        {"id": "gpu0", "gpu": "h100", "node": "node1", "socket": 0}
    )

    with pytest.raises(ValueError, match="duplicate ids"):
        parse_config(config)


def test_build_runtime_rejects_kind_mismatched_preset():
    config = make_canonical_config()
    del config["hardware"]["devices"][0]["gpu"]
    config["hardware"]["devices"][0]["cpu"] = "h100"

    with pytest.raises(ValueError, match="preset usage"):
        build_runtime(parse_config(config))


def test_parse_rejects_memory_placement_for_explicit_stage():
    config = make_canonical_config()
    config["pipeline"]["stages"][0]["memory_placement"] = {
        "kind": "socket",
        "node": "node0",
        "socket": 1,
    }

    with pytest.raises(ValueError, match="only supported for analytical compute mode"):
        parse_config(config)


def test_build_runtime_resolves_interconnect_presets():
    config = make_canonical_config()
    config["pipeline"]["stages"] = [
        {
            "device": "gpu0",
            "weights_mb": 0.0,
            "compute": {"mode": "explicit", "distribution": {"type": "constant", "value": 1.0}},
        },
        {
            "device": "gpu1",
            "weights_mb": 0.0,
            "compute": {"mode": "explicit", "distribution": {"type": "constant", "value": 1.0}},
        },
    ]
    config["hardware"]["devices"] = [
        {"id": "gpu0", "gpu": "a100", "node": "node0", "socket": 0},
        {"id": "gpu1", "gpu": "a100", "node": "node0", "socket": 1},
    ]
    runtime = build_runtime(parse_config(config))

    link = runtime.pipeline.topology.link("gpu0", "gpu1")
    assert link.bandwidth_gbps == 4800.0
    assert link.base_latency_us == 1.0


def test_device_override_takes_precedence_over_preset():
    config = make_canonical_config()
    config["overrides"] = {"devices": [{"id": "gpu0", "memory_mb": 12345.0}]}

    runtime = build_runtime(parse_config(config))

    assert runtime.pipeline.topology.device("gpu0").memory_mb == 12345.0


def test_preset_based_heterogeneous_4gpu_config_runs():
    config = make_canonical_config(batches=1, microbatches=4)
    config["pipeline"]["schedule"] = "1f1b"
    config["pipeline"]["activation_mb"] = 32.0
    config["pipeline"]["stages"] = [
        {
            "device": "gpu0",
            "weights_mb": 1024.0,
            "compute": {"mode": "analytical", "tflop": 4.0, "memory_mb": 64.0},
        },
        {
            "device": "gpu1",
            "weights_mb": 1024.0,
            "compute": {"mode": "analytical", "tflop": 4.0, "memory_mb": 64.0},
        },
        {
            "device": "gpu2",
            "weights_mb": 1536.0,
            "compute": {"mode": "analytical", "tflop": 6.0, "memory_mb": 96.0},
        },
        {
            "device": "gpu3",
            "weights_mb": 1536.0,
            "compute": {"mode": "analytical", "tflop": 6.0, "memory_mb": 96.0},
        },
    ]
    config["hardware"]["devices"] = [
        {"id": "gpu0", "gpu": "h100", "node": "node0", "socket": 0},
        {"id": "gpu1", "gpu": "h100", "node": "node0", "socket": 0},
        {"id": "gpu2", "gpu": "a100", "node": "node1", "socket": 0},
        {"id": "gpu3", "gpu": "a100", "node": "node1", "socket": 0},
    ]
    runtime = build_runtime(parse_config(config))

    runtime.pipeline.start_batch(runtime.num_microbatches)
    runtime.engine.run()

    assert runtime.collector.completed_microbatches == 4
    assert runtime.collector.throughput() > 0
