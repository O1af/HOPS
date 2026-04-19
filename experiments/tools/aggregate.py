"""Aggregate scoring for HOPS validation fixtures.

Reads comparison.json documents produced by compare_run.py and computes
per-variant accuracy metrics used for regression detection.

Tracked metrics per variant:
  - throughput_error_pct: (hops - megatron) / megatron * 100
  - bubble_pp_delta: hops_bubble - megatron_bubble (percentage points)
  - util_spearman_rho: rank correlation of per-stage utilization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


SCORED_VARIANTS = ("no_lookahead", "link_calibrated")


@dataclass(frozen=True)
class VariantScore:
    variant: str
    throughput_per_s: float | None
    throughput_error_pct: float | None
    latency_mean_ms: float | None
    bubble_ratio: float | None
    bubble_pp_delta: float | None
    comm_overhead_ratio: float | None
    comm_overhead_delta: float | None
    util_spearman_rho: float | None
    util_max_abs_delta: float | None
    per_stage_util: list[dict] | None


@dataclass(frozen=True)
class MegatronBaseline:
    throughput_per_s: float | None
    latency_mean_ms: float | None
    bubble_ratio: float | None
    completed_microbatches: int | None


@dataclass(frozen=True)
class FixtureResult:
    fixture_id: str
    megatron: MegatronBaseline
    variant_scores: dict[str, VariantScore]


@dataclass(frozen=True)
class SuiteAggregates:
    throughput_mape: float
    bubble_mae_pp: float
    util_spearman_mean: float | None


@dataclass(frozen=True)
class SuiteResult:
    fixtures: list[FixtureResult]
    aggregates: SuiteAggregates


def _parse_megatron(doc: dict) -> MegatronBaseline:
    m = doc.get("megatron") or {}
    return MegatronBaseline(
        throughput_per_s=m.get("throughput_per_s"),
        latency_mean_ms=m.get("latency_mean_ms"),
        bubble_ratio=m.get("bubble_ratio"),
        completed_microbatches=m.get("completed_microbatches"),
    )


def _parse_variant(name: str, row: dict) -> VariantScore:
    if row.get("status") == "missing":
        return VariantScore(
            variant=name, throughput_per_s=None, throughput_error_pct=None,
            latency_mean_ms=None, bubble_ratio=None, bubble_pp_delta=None,
            comm_overhead_ratio=None, comm_overhead_delta=None,
            util_spearman_rho=None, util_max_abs_delta=None, per_stage_util=None,
        )
    return VariantScore(
        variant=name,
        throughput_per_s=row.get("throughput_per_s"),
        throughput_error_pct=row.get("throughput_error_pct"),
        latency_mean_ms=row.get("latency_mean_ms"),
        bubble_ratio=row.get("bubble_ratio"),
        bubble_pp_delta=row.get("bubble_pp_delta"),
        comm_overhead_ratio=row.get("comm_overhead_ratio"),
        comm_overhead_delta=row.get("comm_overhead_delta"),
        util_spearman_rho=row.get("util_spearman_rho"),
        util_max_abs_delta=row.get("util_max_abs_delta"),
        per_stage_util=row.get("per_stage_util"),
    )


def score_comparison(fixture_id: str, comparison_path: Path) -> FixtureResult:
    """Extract per-variant scores from a comparison.json document."""
    doc = json.loads(comparison_path.read_text(encoding="utf-8"))
    variants = doc.get("variants", {})

    megatron = _parse_megatron(doc)
    scores: dict[str, VariantScore] = {}
    for name in SCORED_VARIANTS:
        scores[name] = _parse_variant(name, variants.get(name, {}))

    return FixtureResult(fixture_id=fixture_id, megatron=megatron, variant_scores=scores)


def compute_suite_score(results: list[FixtureResult]) -> SuiteResult:
    """Compute aggregate metrics across scored variants and fixtures."""
    throughput_vals: list[float] = []
    bubble_vals: list[float] = []
    spearman_vals: list[float] = []

    for r in results:
        for name in SCORED_VARIANTS:
            score = r.variant_scores.get(name)
            if score is None:
                continue
            if score.throughput_error_pct is not None:
                throughput_vals.append(abs(score.throughput_error_pct))
            if score.bubble_pp_delta is not None:
                bubble_vals.append(abs(score.bubble_pp_delta))
            if score.util_spearman_rho is not None:
                spearman_vals.append(score.util_spearman_rho)

    agg = SuiteAggregates(
        throughput_mape=sum(throughput_vals) / len(throughput_vals) if throughput_vals else 0.0,
        bubble_mae_pp=sum(bubble_vals) / len(bubble_vals) if bubble_vals else 0.0,
        util_spearman_mean=sum(spearman_vals) / len(spearman_vals) if spearman_vals else None,
    )
    return SuiteResult(fixtures=results, aggregates=agg)
