"""Fit per-stage compute distributions from a Megatron trace dir.

Reads <job_dir>/megatron_trace/*.jsonl (as emitted by HOPS trace hooks inside
Megatron-LM) and produces a pipeline.stages overlay with one explicit
distribution per stage for the FORWARD phase and, when backward events are
present, one explicit BACKWARD distribution per stage. When the backward
overlay is provided, HOPS uses it directly and ignores `pipeline.backward_factor`.

Warmup iterations are stripped from the trace by the importer using a
3x-trailing-median heuristic; pass --min-iteration to override with a hard cut.

Usage:
    uv run python experiments/tools/derive_megatron_stats.py \\
        --job-dir experiments/experiment_1/1_all_nodes/output/12 \\
        --output   experiments/experiment_1/1_all_nodes/output/12/derived/stage_timings.yaml
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

from _common import ensure_hops_importable, write_generated_yaml


@dataclass(frozen=True)
class StageFit:
    stage: int
    count: int
    mean_ms: float
    std_ms: float


def fit_stage_distributions(events, phase, min_iteration: int = 0) -> list[StageFit]:
    per_stage: dict[int, list[float]] = {}
    for event in events:
        if event.event_type != "compute":
            continue
        if event.phase != phase:
            continue
        if event.iteration < min_iteration:
            continue
        duration_ms = (event.end_wall_ns - event.start_wall_ns) / 1_000_000.0
        per_stage.setdefault(int(event.stage), []).append(float(duration_ms))

    fits: list[StageFit] = []
    for stage, samples in sorted(per_stage.items()):
        if not samples:
            continue
        mean_ms = statistics.mean(samples)
        std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        fits.append(StageFit(stage=stage, count=len(samples), mean_ms=mean_ms, std_ms=std_ms))
    return fits


@dataclass(frozen=True)
class OptimizerFit:
    count: int
    mean_ms: float
    std_ms: float


def fit_optimizer_distribution(events, phase, min_iteration: int = 0) -> OptimizerFit | None:
    """Pool OPTIMIZER compute events across all ranks into one distribution.

    HOPS samples one distribution per iteration for every device, so a single
    pooled fit is sufficient even when per-rank means differ slightly.
    """
    samples: list[float] = []
    for event in events:
        if event.event_type != "compute":
            continue
        if event.phase != phase:
            continue
        if event.iteration < min_iteration:
            continue
        duration_ms = (event.end_wall_ns - event.start_wall_ns) / 1_000_000.0
        samples.append(float(duration_ms))
    if not samples:
        return None
    mean_ms = statistics.mean(samples)
    std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return OptimizerFit(count=len(samples), mean_ms=mean_ms, std_ms=std_ms)


def build_optimizer_overlay(fit: OptimizerFit) -> dict:
    return {
        "optimizer": {
            "enabled": True,
            "update": _normal_distribution(fit),
        }
    }


@dataclass(frozen=True)
class BarrierFit:
    count: int
    mean_ms: float
    std_ms: float


def fit_iteration_barrier(events, min_iteration: int = 0) -> BarrierFit | None:
    """Fit per-iteration host/framework dead time.

    For each iteration, the dead time is the wall duration minus the UNION of
    all event intervals (compute + transfer, across all ranks). This captures
    every interval where no rank is doing work the pipeline DAG tracks:
    CPU-side overhead, stream sync gaps between backward and optimizer,
    framework stalls, data loading, inter-iteration barriers, etc.

    The per-iter wall extends from the iteration's first event start to the
    NEXT iteration's first event start, so inter-iteration gaps are folded in
    as well. Negative or zero values are dropped.
    """
    per_iter_intervals: dict[int, list[tuple[int, int]]] = {}
    per_iter_start: dict[int, int] = {}
    for event in events:
        if event.iteration < min_iteration:
            continue
        it = int(event.iteration)
        per_iter_intervals.setdefault(it, []).append(
            (event.start_wall_ns, event.end_wall_ns)
        )
        if it not in per_iter_start or event.start_wall_ns < per_iter_start[it]:
            per_iter_start[it] = event.start_wall_ns

    iters = sorted(per_iter_start)
    if len(iters) < 2:
        return None

    dead_ms: list[float] = []
    for a, b in zip(iters, iters[1:]):
        intervals = sorted(per_iter_intervals[a])
        if not intervals:
            continue
        wall_start = per_iter_start[a]
        wall_end = per_iter_start[b]
        if wall_end <= wall_start:
            continue
        union_ns = 0
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                union_ns += cur_e - cur_s
                cur_s, cur_e = s, e
        union_ns += cur_e - cur_s
        dead_ns = max(0, (wall_end - wall_start) - union_ns)
        dead_ms.append(dead_ns / 1_000_000.0)

    if not dead_ms:
        return None
    mean_ms = statistics.mean(dead_ms)
    std_ms = statistics.pstdev(dead_ms) if len(dead_ms) > 1 else 0.0
    return BarrierFit(count=len(dead_ms), mean_ms=mean_ms, std_ms=std_ms)


def build_iteration_barrier_overlay(fit: BarrierFit) -> dict:
    return {"optimizer": {"iteration_barrier": _normal_distribution(fit)}}


def _normal_distribution(fit) -> dict:
    return {"type": "normal", "mean": round(fit.mean_ms, 4), "std": round(fit.std_ms, 4)}


def build_overlay(forward_fits: list[StageFit],
                  backward_fits: list[StageFit] | None = None) -> dict:
    backward_by_stage = {fit.stage: fit for fit in (backward_fits or [])}
    stages: list[dict] = []
    for fit in forward_fits:
        stage_entry: dict = {
            "id": fit.stage,
            "compute": {"mode": "explicit", "distribution": _normal_distribution(fit)},
        }
        bwd = backward_by_stage.get(fit.stage)
        if bwd is not None:
            stage_entry["backward"] = {"distribution": _normal_distribution(bwd)}
        stages.append(stage_entry)
    return {"pipeline": {"stages": stages}}


def _write_optional_overlay(path, fit, build_overlay, trace_dir: Path, source_note: str) -> None:
    if path is None or fit is None:
        return
    banner = (
        "# GENERATED by experiments/tools/derive_megatron_stats.py. DO NOT EDIT.\n"
        f"# Source: {trace_dir} ({source_note}).\n"
    )
    write_generated_yaml(path, banner, build_overlay(fit))


def write_stage_timings_overlay(
    trace_dir: Path,
    output_path: Path,
    min_iteration: int = 0,
    optimizer_output_path: Path | None = None,
    iteration_barrier_output_path: Path | None = None,
) -> tuple[list[StageFit], list[StageFit], OptimizerFit | None, BarrierFit | None]:
    ensure_hops_importable()
    from hops.core.types import Phase  # type: ignore[import-not-found]
    from hops.megatron.importer import load_raw_megatron_events  # type: ignore[import-not-found]

    # When the caller pins min_iteration, take that as ground truth and skip
    # the heuristic; otherwise let the importer strip warmup automatically.
    strip = min_iteration == 0
    events = load_raw_megatron_events(trace_dir, strip_warmup=strip)
    forward_fits = fit_stage_distributions(
        events, phase=Phase.FORWARD, min_iteration=min_iteration
    )
    if not forward_fits:
        return [], [], None, None
    backward_fits = fit_stage_distributions(
        events, phase=Phase.BACKWARD, min_iteration=min_iteration
    )
    optimizer_fit = fit_optimizer_distribution(
        events, phase=Phase.OPTIMIZER, min_iteration=min_iteration
    )
    barrier_fit = fit_iteration_barrier(events, min_iteration=min_iteration)

    backward_note = (
        "Forward + BACKWARD compute events fit per stage; HOPS ignores backward_factor for these stages."
        if backward_fits
        else "FORWARD compute events only; backward times are derived via pipeline.backward_factor."
    )
    banner = (
        "# GENERATED by experiments/tools/derive_megatron_stats.py. DO NOT EDIT.\n"
        f"# Source: {trace_dir}.\n"
        f"# {backward_note}\n"
    )
    write_generated_yaml(output_path, banner, build_overlay(forward_fits, backward_fits))

    _write_optional_overlay(
        optimizer_output_path,
        optimizer_fit,
        build_optimizer_overlay,
        trace_dir,
        "OPTIMIZER compute events, pooled across ranks",
    )
    _write_optional_overlay(
        iteration_barrier_output_path,
        barrier_fit,
        build_iteration_barrier_overlay,
        trace_dir,
        "inter-iteration wall-clock gaps, pooled across adjacent iteration pairs",
    )

    return forward_fits, backward_fits, optimizer_fit, barrier_fit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, help="Path to output/<job-id>")
    parser.add_argument("--output", required=True, help="Path to write stage_timings.yaml")
    parser.add_argument(
        "--min-iteration",
        type=int,
        default=0,
        help=(
            "Skip events with iteration < this value. Defaults to 0, in which "
            "case the importer's 3x-trailing-median heuristic strips warmup."
        ),
    )
    parser.add_argument(
        "--optimizer-output",
        default=None,
        help="If set, also write a fitted optimizer.yaml overlay to this path",
    )
    parser.add_argument(
        "--iteration-barrier-output",
        default=None,
        help="If set, also write a fitted iteration_barrier.yaml overlay to this path",
    )
    return parser.parse_args()


def _format_fits(label: str, fits: list[StageFit]) -> str:
    if not fits:
        return f"{label}: none"
    return label + ": " + ", ".join(
        f"stage {f.stage} n={f.count} mean={f.mean_ms:.2f}ms std={f.std_ms:.2f}ms"
        for f in fits
    )


def main() -> None:
    args = _parse_args()
    job_dir = Path(args.job_dir)
    trace_dir = job_dir / "megatron_trace"
    if not trace_dir.is_dir():
        raise SystemExit(f"megatron_trace dir not found: {trace_dir}")

    output_path = Path(args.output)
    optimizer_output = Path(args.optimizer_output) if args.optimizer_output else None
    barrier_output = (
        Path(args.iteration_barrier_output) if args.iteration_barrier_output else None
    )
    forward_fits, backward_fits, optimizer_fit, barrier_fit = write_stage_timings_overlay(
        trace_dir, output_path, args.min_iteration, optimizer_output, barrier_output
    )
    if not forward_fits:
        raise SystemExit("no FORWARD compute events found in trace")

    print(f"wrote {len(forward_fits)} stage fits -> {output_path}")
    print(_format_fits("forward", forward_fits))
    print(_format_fits("backward", backward_fits))
    if optimizer_fit is not None and optimizer_output is not None:
        print(
            f"optimizer: n={optimizer_fit.count} mean={optimizer_fit.mean_ms:.2f}ms "
            f"std={optimizer_fit.std_ms:.2f}ms -> {optimizer_output}"
        )
    elif optimizer_output is not None:
        print(f"optimizer: no OPTIMIZER events found in trace; {optimizer_output} not written")
    if barrier_fit is not None and barrier_output is not None:
        print(
            f"iteration_barrier: n={barrier_fit.count} mean={barrier_fit.mean_ms:.2f}ms "
            f"std={barrier_fit.std_ms:.2f}ms -> {barrier_output}"
        )
    elif barrier_output is not None:
        print(
            f"iteration_barrier: insufficient iterations in trace; "
            f"{barrier_output} not written"
        )


if __name__ == "__main__":
    main()
