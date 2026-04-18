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
from pathlib import Path

from _common import ensure_hops_importable


VARIANT_ORDER = ("no_lookahead", "link_calibrated", "trace_replay")

VARIANT_DESCRIPTIONS = {
    "no_lookahead": "committed hops.base.yaml only; no run lookahead",
    "link_calibrated": (
        "adds measured links + framework-level overlays (iteration_barrier, "
        "optimizer kernel distribution). Compute still analytical. Overlays are "
        "cluster/framework properties reusable across model shapes."
    ),
    "trace_replay": (
        "posthoc diagnostic; adds per-stage compute distribution from this "
        "run's megatron_trace on top of link_calibrated. Overfit by construction."
    ),
}

VARIANT_SOURCES = {
    "no_lookahead": "analytical tflop + default efficiency",
    "link_calibrated": (
        "analytical tflop + default efficiency; framework-level iteration_barrier "
        "and optimizer kernel distribution fit from trace"
    ),
    "trace_replay": (
        "per-stage forward + backward distribution fit from megatron_trace; "
        "optimizer step distribution pooled across ranks when OPTIMIZER events present"
    ),
}


def _import_hops():
    ensure_hops_importable()
    from hops.megatron.compare import build_comparison, convert_job_dir  # type: ignore[import-not-found]
    return build_comparison, convert_job_dir


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_megatron_summary(job_dir: Path, convert_job_dir) -> dict | None:
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

    convert_job_dir(job_dir)
    return _load_json(summary_path)


def _pct_delta(actual: float | None, predicted: float | None) -> float | None:
    if actual is None or predicted is None or actual == 0:
        return None
    return (predicted - actual) / actual * 100.0


def _summary_scalar(summary: dict, *keys: str) -> float | None:
    cursor: object = summary
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    if isinstance(cursor, (int, float)):
        return float(cursor)
    return None


def _spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation of two equal-length sequences. Returns None if degenerate."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def _ranks(values: list[float]) -> list[float]:
        # Average ranks for ties. Ranks are 1-based.
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denom_x = sum((rx[i] - mx) ** 2 for i in range(n))
    denom_y = sum((ry[i] - my) ** 2 for i in range(n))
    if denom_x == 0 or denom_y == 0:
        return None
    return num / (denom_x * denom_y) ** 0.5


def _per_stage_util_arrays(comparison: dict) -> tuple[list[str], list[float], list[float]] | None:
    per_stage = comparison.get("utilization", {}).get("per_stage")
    if not isinstance(per_stage, dict) or not per_stage:
        return None
    keys = sorted(per_stage, key=lambda k: int(k))
    mega = []
    hops = []
    for key in keys:
        entry = per_stage[key]
        m = entry.get("megatron")
        h = entry.get("hops")
        if m is None or h is None:
            return None
        mega.append(float(m))
        hops.append(float(h))
    return keys, mega, hops


def _variant_row(summary: dict, megatron: dict | None, build_comparison) -> dict:
    hops_throughput = _summary_scalar(summary, "throughput", "per_s")
    megatron_throughput = _summary_scalar(megatron, "throughput", "per_s") if megatron else None
    comparison = build_comparison(megatron, summary) if megatron is not None else None

    bubble_delta = None
    bubble_pp_delta = None
    comm_overhead_delta = None
    util_spearman_rho = None
    per_stage_util = None
    util_max_abs_delta = None
    if comparison is not None:
        bubble = comparison.get("bubble_ratio", {})
        # Report (sim - real) so the sign matches the reader's intuition:
        # positive = HOPS over-predicts idleness, negative = HOPS under-predicts.
        mega_b = bubble.get("megatron")
        hops_b = bubble.get("hops")
        if mega_b is not None and hops_b is not None:
            bubble_delta = hops_b - mega_b
            bubble_pp_delta = bubble_delta * 100.0
        comm = comparison.get("communication_overhead_ratio", {})
        mega_c = comm.get("megatron")
        hops_c = comm.get("hops")
        if mega_c is not None and hops_c is not None:
            comm_overhead_delta = hops_c - mega_c

        arrays = _per_stage_util_arrays(comparison)
        if arrays is not None:
            keys, mega_u, hops_u = arrays
            util_spearman_rho = _spearman_rho(mega_u, hops_u)
            per_stage_util = [
                {
                    "stage": key,
                    "megatron": mega_u[i],
                    "hops": hops_u[i],
                    "delta": hops_u[i] - mega_u[i],
                }
                for i, key in enumerate(keys)
            ]
            util_max_abs_delta = max(
                abs(hops_u[i] - mega_u[i]) for i in range(len(keys))
            )

    return {
        "summary": summary,
        "comparison": comparison,
        "throughput_per_s": hops_throughput,
        "throughput_error_pct": _pct_delta(megatron_throughput, hops_throughput),
        "latency_mean_ms": _summary_scalar(summary, "latency_ms", "mean_ms"),
        "bubble_ratio": _summary_scalar(summary, "bubble_ratio"),
        "bubble_delta": bubble_delta,
        "bubble_pp_delta": bubble_pp_delta,
        "comm_overhead_ratio": _summary_scalar(summary, "time_ms", "communication_overhead_ratio"),
        "comm_overhead_delta": comm_overhead_delta,
        "util_spearman_rho": util_spearman_rho,
        "util_max_abs_delta": util_max_abs_delta,
        "per_stage_util": per_stage_util,
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
        else:
            variants[name] = _variant_row(summary, megatron, build_comparison)

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
    lines.append(
        "| variant | throughput_per_s | throughput error (%) | "
        "bubble_ratio | bubble Δ (pp, sim−real) | "
        "comm_overhead Δ (sim−real) | util max |Δ| | util Spearman ρ |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for name in VARIANT_ORDER:
        row = variants.get(name, {"status": "missing"})
        if row.get("status") == "missing":
            lines.append(f"| {name} | missing | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {format_number(row.get('throughput_per_s'))} | "
            f"{format_number(row.get('throughput_error_pct'))} | "
            f"{format_number(row.get('bubble_ratio'))} | "
            f"{format_number(row.get('bubble_pp_delta'))} | "
            f"{format_number(row.get('comm_overhead_delta'), '.4f')} | "
            f"{format_number(row.get('util_max_abs_delta'), '.4f')} | "
            f"{format_number(row.get('util_spearman_rho'))} |"
        )
    lines.append("")

    # Per-stage util table(s) — show the shape of the error, not just the scalar.
    for name in VARIANT_ORDER:
        row = variants.get(name, {})
        per_stage = row.get("per_stage_util") if isinstance(row, dict) else None
        if not per_stage:
            continue
        lines.append(f"### {name} — per-stage utilization")
        lines.append("")
        lines.append("| stage | real | sim | Δ (sim−real) |")
        lines.append("| --- | --- | --- | --- |")
        for entry in per_stage:
            lines.append(
                f"| {entry['stage']} | "
                f"{format_number(entry['megatron'])} | "
                f"{format_number(entry['hops'])} | "
                f"{format_number(entry['delta'])} |"
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

    lines.append("> **Note on transfer\\_ms**: Megatron `transfer_ms` reflects host-side NCCL")
    lines.append("> dispatch time (async enqueue). HOPS `transfer_ms` models full wire transfer")
    lines.append("> duration. These metrics are not directly comparable. Use throughput and")
    lines.append("> latency for accuracy assessment.")
    lines.append("")
    lines.append("## Overfit guard")
    for name in VARIANT_ORDER:
        lines.append(f"- {name}: {document['overfit_guard'][name]}")
    lines.append("")

    return "\n".join(lines)


def run(job_dir: Path, links_source: str = "(unknown)") -> tuple[Path, Path]:
    derived_dir = job_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    build_comparison, convert_job_dir = _import_hops()
    megatron = _ensure_megatron_summary(job_dir, convert_job_dir)

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
