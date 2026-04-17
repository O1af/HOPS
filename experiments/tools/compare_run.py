"""Compare HOPS variants against the real Megatron run for one job.

Reads summary JSONs under <job-dir>/ (megatron_summary.json) and
<job-dir>/derived/ (hops_no_lookahead_summary.json,
hops_link_calibrated_summary.json, hops_trace_replay_summary.json) and emits
<job-dir>/derived/comparison.json plus a concise markdown report at
<job-dir>/derived/report.md.

If megatron_summary.json is missing but <job-dir>/megatron_trace/*.jsonl is
present, it runs hops.megatron.compare.convert_job_dir to produce it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


VARIANT_ORDER = ("no_lookahead", "link_calibrated", "trace_replay")

VARIANT_DESCRIPTIONS = {
    "no_lookahead": "committed hops.base.yaml only; no run lookahead",
    "link_calibrated": "adds measured links; compute still analytical",
    "trace_replay": "posthoc; stage compute fit from megatron_trace",
}

VARIANT_SOURCES = {
    "no_lookahead": "analytical tflop + default efficiency",
    "link_calibrated": "analytical tflop + default efficiency",
    "trace_replay": "per-stage forward distribution fit from megatron_trace",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_hops():
    sys.path.insert(0, str(_repo_root() / "src"))
    from hops.megatron.compare import build_comparison, convert_job_dir  # type: ignore[import-not-found]
    return build_comparison, convert_job_dir


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_megatron_summary(job_dir: Path) -> dict | None:
    summary_path = job_dir / "megatron_summary.json"
    existing = _load_json(summary_path)
    if existing is not None:
        return existing

    trace_dir = job_dir / "megatron_trace"
    if not trace_dir.is_dir():
        return None
    try:
        next(trace_dir.glob("*.jsonl"))
    except StopIteration:
        return None

    _, convert_job_dir = _import_hops()
    convert_job_dir(job_dir)
    return _load_json(summary_path)


def _pct_delta(actual: float | None, predicted: float | None) -> float | None:
    """Error of HOPS prediction vs measured Megatron, in percent."""
    if actual is None or predicted is None:
        return None
    if actual == 0:
        return None
    return (predicted - actual) / actual * 100.0


def _summary_scalar(summary: dict, *keys: str) -> float | None:
    cursor: Any = summary
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, (int, float)):
        return float(cursor)
    return None


def build_variant_row(megatron: dict, hops_summary: dict, build_comparison) -> dict:
    return {
        "summary": hops_summary,
        "comparison": build_comparison(megatron, hops_summary),
        "throughput_per_s": _summary_scalar(hops_summary, "throughput", "per_s"),
        "throughput_error_pct": _pct_delta(
            _summary_scalar(megatron, "throughput", "per_s"),
            _summary_scalar(hops_summary, "throughput", "per_s"),
        ),
        "latency_mean_ms": _summary_scalar(hops_summary, "latency_ms", "mean_ms"),
        "bubble_ratio": _summary_scalar(hops_summary, "bubble_ratio"),
    }


def build_comparison_document(
    job_id: str,
    megatron: dict | None,
    variant_summaries: dict[str, dict],
    build_comparison,
    links_source: str,
) -> dict:
    variants: dict[str, dict] = {}
    for name in VARIANT_ORDER:
        summary = variant_summaries.get(name)
        if summary is None:
            variants[name] = {"status": "missing"}
            continue
        if megatron is None:
            variants[name] = {
                "summary": summary,
                "comparison": None,
                "throughput_per_s": _summary_scalar(summary, "throughput", "per_s"),
                "throughput_error_pct": None,
                "latency_mean_ms": _summary_scalar(summary, "latency_ms", "mean_ms"),
                "bubble_ratio": _summary_scalar(summary, "bubble_ratio"),
            }
        else:
            variants[name] = build_variant_row(megatron, summary, build_comparison)

    megatron_headline = None
    if megatron is not None:
        megatron_headline = {
            "throughput_per_s": _summary_scalar(megatron, "throughput", "per_s"),
            "latency_mean_ms": _summary_scalar(megatron, "latency_ms", "mean_ms"),
            "bubble_ratio": _summary_scalar(megatron, "bubble_ratio"),
            "completed_microbatches": megatron.get("completed_microbatches"),
        }

    return {
        "job_id": job_id,
        "megatron": megatron_headline,
        "variants": variants,
        "calibration_manifest": {
            "activation_source": "derived from pipeline.model (hidden_dim, seq_len)",
            "links_source": links_source,
            "compute_source_per_variant": dict(VARIANT_SOURCES),
        },
        "overfit_guard": dict(VARIANT_DESCRIPTIONS),
    }


def format_number(value: float | None, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    return format(value, spec)


def build_markdown_report(document: dict) -> str:
    megatron = document["megatron"]
    variants = document["variants"]
    job_id = document["job_id"]

    lines: list[str] = []
    lines.append(f"# Run {job_id}")
    lines.append("")
    lines.append("## Real Megatron")
    if megatron is None:
        lines.append("- not available (no megatron_trace found)")
    else:
        lines.append(f"- throughput_per_s: {format_number(megatron.get('throughput_per_s'))}")
        lines.append(f"- latency_mean_ms: {format_number(megatron.get('latency_mean_ms'))}")
        lines.append(f"- bubble_ratio: {format_number(megatron.get('bubble_ratio'))}")
    lines.append("")

    lines.append("## HOPS variants")
    lines.append("")
    lines.append("| variant | throughput_per_s | error vs megatron (%) | latency_mean_ms | bubble_ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name in VARIANT_ORDER:
        row = variants.get(name, {"status": "missing"})
        if row.get("status") == "missing":
            lines.append(f"| {name} | missing | — | — | — |")
            continue
        lines.append(
            f"| {name} | {format_number(row.get('throughput_per_s'))} | "
            f"{format_number(row.get('throughput_error_pct'))} | "
            f"{format_number(row.get('latency_mean_ms'))} | "
            f"{format_number(row.get('bubble_ratio'))} |"
        )
    lines.append("")

    manifest = document["calibration_manifest"]
    lines.append("## Calibration used")
    lines.append(f"- activation: {manifest['activation_source']}")
    lines.append(f"- links: {manifest['links_source']}")
    lines.append("- compute:")
    for name in VARIANT_ORDER:
        lines.append(f"    - {name}: {manifest['compute_source_per_variant'][name]}")
    lines.append("")

    lines.append("## Overfit guard")
    for name in VARIANT_ORDER:
        lines.append(f"- {name}: {document['overfit_guard'][name]}")
    lines.append("")

    return "\n".join(lines)


def run(job_dir: Path, links_source: str = "(unknown)") -> tuple[Path, Path]:
    derived_dir = job_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    build_comparison, _ = _import_hops()
    megatron = _ensure_megatron_summary(job_dir)

    variant_summaries: dict[str, dict] = {}
    for name in VARIANT_ORDER:
        summary = _load_json(derived_dir / f"hops_{name}_summary.json")
        if summary is not None:
            variant_summaries[name] = summary

    document = build_comparison_document(
        job_id=job_dir.name,
        megatron=megatron,
        variant_summaries=variant_summaries,
        build_comparison=build_comparison,
        links_source=links_source,
    )

    comparison_path = derived_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)

    report_path = derived_dir / "report.md"
    report_path.write_text(build_markdown_report(document), encoding="utf-8")

    return comparison_path, report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Path to output/<job-id>")
    parser.add_argument(
        "--links-source",
        default="(unknown)",
        help="Human-readable note about where measured links came from",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    comparison_path, report_path = run(Path(args.job_dir), links_source=args.links_source)
    print(f"wrote {comparison_path} and {report_path}")
    print()
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
