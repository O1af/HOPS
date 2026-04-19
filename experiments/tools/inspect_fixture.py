"""Inspect a train fixture's real Megatron trace data.

Prints per-stage compute timing, per-link transfer stats, and iteration
barrier estimates — the raw ground truth the agent needs during the OBSERVE
phase to form hypotheses about simulator error sources.

Usage:
    uv run python experiments/tools/inspect_fixture.py --fixture exp2_1_all_nodes_run51
    uv run python experiments/tools/inspect_fixture.py --fixture exp2_13_a10g_pair_pp2_run135 --phase BACKWARD
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from _common import ensure_hops_importable, repo_root

sys.path.insert(0, str(repo_root() / "fixtures"))
from loader import discover_fixtures, materialize_fixture  # noqa: E402

ensure_hops_importable()
from hops.core.types import Phase  # noqa: E402
from hops.megatron.importer import load_raw_megatron_events  # noqa: E402


def _print_compute_stats(events, phases=None):
    per_key: dict[tuple[int, str], list[float]] = defaultdict(list)
    for e in events:
        if e.event_type != "compute":
            continue
        if phases and e.phase not in phases:
            continue
        dur_ms = (e.end_wall_ns - e.start_wall_ns) / 1_000_000.0
        per_key[(e.stage, e.phase.name)].append(dur_ms)

    if not per_key:
        print("  (no compute events)")
        return

    print(f"  {'stage':<6s} {'phase':<12s} {'count':>5s}  {'mean_ms':>9s}  {'std_ms':>8s}  {'min_ms':>8s}  {'max_ms':>8s}")
    print("  " + "-" * 60)
    for (stage, phase_name) in sorted(per_key):
        samples = per_key[(stage, phase_name)]
        mean = statistics.mean(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        print(
            f"  {stage:<6d} {phase_name:<12s} {len(samples):>5d}  "
            f"{mean:>9.3f}  {std:>8.3f}  {min(samples):>8.3f}  {max(samples):>8.3f}"
        )


def _print_transfer_stats(events):
    per_link: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e.event_type != "transfer":
            continue
        key = f"{e.src_device} -> {e.dst_device}"
        dur_ms = (e.end_wall_ns - e.start_wall_ns) / 1_000_000.0
        per_link[key].append(dur_ms)

    if not per_link:
        print("  (no transfer events)")
        return

    print(f"  {'link':<45s} {'count':>5s}  {'mean_ms':>9s}  {'std_ms':>8s}  {'min_ms':>8s}  {'max_ms':>8s}")
    print("  " + "-" * 85)
    for link in sorted(per_link):
        samples = per_link[link]
        mean = statistics.mean(samples)
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        print(
            f"  {link:<45s} {len(samples):>5d}  "
            f"{mean:>9.3f}  {std:>8.3f}  {min(samples):>8.3f}  {max(samples):>8.3f}"
        )


def _print_iteration_stats(events):
    per_iter: dict[int, list] = defaultdict(list)
    for e in events:
        per_iter[e.iteration].append(e)

    if len(per_iter) < 2:
        print("  (insufficient iterations)")
        return

    iters = sorted(per_iter)
    wall_durations: list[float] = []
    for a, b in zip(iters, iters[1:]):
        start = min(e.start_wall_ns for e in per_iter[a])
        end = min(e.start_wall_ns for e in per_iter[b])
        wall_durations.append((end - start) / 1_000_000.0)

    print(f"  iterations: {len(iters)} (after warmup strip)")
    print(f"  iter wall time: mean={statistics.mean(wall_durations):.2f}ms  "
          f"std={statistics.pstdev(wall_durations):.2f}ms  "
          f"min={min(wall_durations):.2f}ms  max={max(wall_durations):.2f}ms")


def _print_config_summary(fixture_dir: Path):
    import yaml
    base_path = fixture_dir / "hops.base.yaml"
    if not base_path.exists():
        return
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    pipeline = cfg.get("pipeline", {})
    sim = cfg.get("simulation", {})
    model = pipeline.get("model", {})
    stages = pipeline.get("stages", [])
    devices = [s.get("device", "?").split("_")[0] for s in stages]
    device_counts: dict[str, int] = {}
    for d in devices:
        device_counts[d] = device_counts.get(d, 0) + 1
    device_str = " + ".join(f"{v}x {k}" for k, v in device_counts.items())
    print(f"  scheduler: {pipeline.get('schedule', '?')}  precision: {pipeline.get('precision', '?')}  "
          f"backward_factor: {pipeline.get('backward_factor', '?')}")
    print(f"  stages: {len(stages)} ({device_str})")
    print(f"  model: hidden={model.get('hidden_dim', '?')} seq={model.get('seq_len', '?')}  "
          f"batches: {sim.get('batches', '?')}  microbatches: {sim.get('microbatches', '?')}")
    tflops = [s.get("compute", {}).get("tflop", "?") for s in stages]
    if len(set(str(t) for t in tflops)) == 1:
        print(f"  compute: analytical tflop={tflops[0]} (all stages)")
    else:
        print(f"  compute: analytical tflop={tflops}")


def inspect(fixture_dir: Path, phases=None):
    print(f"=== {fixture_dir.name} ===\n")

    print("CONFIG")
    _print_config_summary(fixture_dir)
    print()

    with tempfile.TemporaryDirectory(prefix="hops_inspect_") as tmpdir:
        workdir = Path(tmpdir)
        scenario_dir, job_id = materialize_fixture(fixture_dir, workdir)
        trace_dir = scenario_dir / "output" / job_id / "megatron_trace"
        events = load_raw_megatron_events(trace_dir)

    print(f"Events: {len(events)} (warmup stripped)\n")

    print("COMPUTE TIMING")
    _print_compute_stats(events, phases)

    print("\nTRANSFER TIMING")
    _print_transfer_stats(events)

    print("\nITERATION TIMING")
    _print_iteration_stats(events)
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", required=True, help="Fixture directory name")
    parser.add_argument("--phase", nargs="*", default=None,
                        help="Filter compute to these phases (e.g. FORWARD BACKWARD)")
    parser.add_argument("--split", default="all",
                        help="Which split to search for the fixture (default: all)")
    args = parser.parse_args()

    phases = None
    if args.phase:
        phases = set()
        for p in args.phase:
            try:
                phases.add(Phase[p.upper()])
            except KeyError:
                raise SystemExit(f"Unknown phase: {p}. Valid: {sorted(Phase.__members__)}")

    all_fixtures = discover_fixtures(split=args.split)
    matches = [f for f in all_fixtures if f.name == args.fixture]
    if not matches:
        raise SystemExit(f"Fixture not found: {args.fixture}")

    inspect(matches[0], phases)


if __name__ == "__main__":
    main()
