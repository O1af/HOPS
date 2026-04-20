"""Tests for heterogeneous pipeline schedulers."""

from __future__ import annotations

from pathlib import Path

import yaml

from hops.config import parse_config
from hops.core.hetero_schedulers import (
    HopsHetero,
    HeteroAdaptiveWSplit,
    HeteroAdaptiveWarmup,
    HeteroBottleneckEagerW,
    HeteroEagerLastW,
    HeteroEagerW,
    HeteroHybrid,
    HeteroFusedBW,
    HeteroCriticalPath,
    HeteroWaveFill,
    HeteroBottleneckPriority,
)
from hops.core.scheduler import _SCHEDULER_REGISTRY, make_scheduler
from hops.runtime import build_runtime


REPO = Path(__file__).resolve().parents[1]
HETERO_CFG_DIR = REPO / "experiments" / "hetero_scheduler"


def _simulate(config_path: Path, scheduler: str) -> float:
    raw = yaml.safe_load(config_path.read_text())
    raw["pipeline"] = {**raw["pipeline"], "schedule": scheduler}
    raw["output"] = {}
    config = parse_config(raw)
    runtime = build_runtime(config)
    for _ in range(runtime.num_batches):
        runtime.pipeline.start_batch(runtime.num_microbatches)
        runtime.engine.run(stop_condition=lambda: runtime.pipeline.batch_complete)
    return runtime.reporter.summary_model().time_ms.makespan_ms


def test_all_hetero_schedulers_registered() -> None:
    for name in [
        "hops_hetero",
        "hetero_adaptive_w_split",
        "hetero_adaptive_warmup",
        "hetero_bottleneck_eager_w",
        "hetero_eager_last_w",
        "hetero_eager_w",
        "hetero_hybrid",
        "hetero_fused_bw",
        "hetero_critical_path",
        "hetero_wavefill",
        "hetero_bottleneck",
    ]:
        assert name in _SCHEDULER_REGISTRY


def test_hops_hetero_matches_zb_when_bottleneck_is_middle() -> None:
    cfg = HETERO_CFG_DIR / "bench_l4_middle_mb24.yaml"
    zb_ms = _simulate(cfg, "zero_bubble")
    hops_ms = _simulate(cfg, "hops_hetero")
    assert hops_ms <= zb_ms * 1.001


def test_hops_hetero_beats_or_matches_zb_when_bottleneck_is_last() -> None:
    cfg = HETERO_CFG_DIR / "bench_a10g_middle_mb48.yaml"
    zb_ms = _simulate(cfg, "zero_bubble")
    hops_ms = _simulate(cfg, "hops_hetero")
    assert hops_ms <= zb_ms * 1.001


def test_hops_hetero_decision_respects_tail_bottleneck() -> None:
    hh = HopsHetero()
    hh.configure({
        "num_stages": 4,
        "fwd_ms": [10.0, 10.0, 10.0, 20.0],
        "b_ms": [5.0, 5.0, 5.0, 10.0],
        "w_ms": [5.0, 5.0, 5.0, 10.0],
    })
    assert hh.uses_w_split is False

    hh2 = HopsHetero()
    hh2.configure({
        "num_stages": 4,
        "fwd_ms": [10.0, 20.0, 20.0, 10.0],
        "b_ms": [5.0, 10.0, 10.0, 5.0],
        "w_ms": [5.0, 10.0, 10.0, 5.0],
    })
    assert hh2.uses_w_split is True


def test_hops_hetero_make_scheduler() -> None:
    assert isinstance(make_scheduler({"policy": "hops_hetero"}), HopsHetero)
