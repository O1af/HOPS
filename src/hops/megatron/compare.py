"""Convert Megatron raw traces into HOPS-compatible artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hops.metrics.exporter import TraceExporter
from hops.metrics.reporter import Reporter
from hops.megatron.importer import import_megatron_trace_dir


@dataclass(frozen=True)
class ConversionOutputs:
    megatron_trace_csv: Path
    megatron_summary_json: Path
    comparison_json: Path | None = None


def _metric_delta(megatron_value: float | None, hops_value: float | None) -> dict[str, float | None]:
    if megatron_value is None or hops_value is None:
        return {
            "megatron": megatron_value,
            "hops": hops_value,
            "delta": None,
            "pct_delta_vs_hops": None,
        }
    delta = megatron_value - hops_value
    pct = None if hops_value == 0 else (delta / hops_value) * 100.0
    return {
        "megatron": megatron_value,
        "hops": hops_value,
        "delta": delta,
        "pct_delta_vs_hops": pct,
    }


def _compare_mapped_metrics(
    megatron_map: dict[str, float],
    hops_map: dict[str, float],
) -> dict[str, dict[str, float | None]]:
    keys = sorted(set(megatron_map) | set(hops_map))
    return {
        key: _metric_delta(megatron_map.get(key), hops_map.get(key))
        for key in keys
    }


def build_comparison(megatron_summary: dict[str, Any], hops_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "throughput": _metric_delta(
            megatron_summary["throughput"]["per_s"],
            hops_summary["throughput"]["per_s"],
        ),
        "latency_ms": {
            "mean_ms": _metric_delta(
                megatron_summary["latency_ms"]["mean_ms"],
                hops_summary["latency_ms"]["mean_ms"],
            ),
            "p50_ms": _metric_delta(
                megatron_summary["latency_ms"]["p50_ms"],
                hops_summary["latency_ms"]["p50_ms"],
            ),
            "p99_ms": _metric_delta(
                megatron_summary["latency_ms"]["p99_ms"],
                hops_summary["latency_ms"]["p99_ms"],
            ),
        },
        "bubble_ratio": _metric_delta(
            megatron_summary["bubble_ratio"],
            hops_summary["bubble_ratio"],
        ),
        "communication_overhead_ratio": _metric_delta(
            megatron_summary["time_ms"]["communication_overhead_ratio"],
            hops_summary["time_ms"]["communication_overhead_ratio"],
        ),
        "utilization": {
            "per_stage": _compare_mapped_metrics(
                {str(k): v for k, v in megatron_summary["utilization"]["per_stage"].items()},
                {str(k): v for k, v in hops_summary["utilization"]["per_stage"].items()},
            ),
            "per_link": _compare_mapped_metrics(
                megatron_summary["utilization"]["per_link"],
                hops_summary["utilization"]["per_link"],
            ),
        },
    }


def convert_job_dir(job_dir: str | Path) -> ConversionOutputs:
    job_path = Path(job_dir)
    trace_dir = job_path / "megatron_trace"
    collector = import_megatron_trace_dir(trace_dir)
    reporter = Reporter(collector)
    summary = reporter.summary_model()

    megatron_trace_csv = job_path / "megatron_trace.csv"
    megatron_summary_json = job_path / "megatron_summary.json"
    TraceExporter(collector).write_csv(str(megatron_trace_csv))
    reporter.write_summary_json(str(megatron_summary_json), summary)

    comparison_json = None
    hops_summary_path = job_path / "hops_summary.json"
    try:
        hops_summary = json.loads(hops_summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass
    else:
        comparison = build_comparison(summary.to_dict(), hops_summary)
        comparison_json = job_path / "comparison.json"
        with comparison_json.open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2, sort_keys=True)

    return ConversionOutputs(
        megatron_trace_csv=megatron_trace_csv,
        megatron_summary_json=megatron_summary_json,
        comparison_json=comparison_json,
    )
