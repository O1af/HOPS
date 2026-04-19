"""Extract a fixture from an experiment's output into fixtures/cluster_results/.

Supports two invocation patterns matching existing experiment layouts:

  # Explicit job IDs (works for any experiment):
  uv run python experiments/tools/extract_fixture.py \
      --scenario experiments/experiment_1/1_all_nodes \
      --run-job-id 15 --link-bench-job-id 16

  # From sequential_jobs.tsv (experiment_2 pattern):
  uv run python experiments/tools/extract_fixture.py \
      --experiment experiments/experiment_2 \
      --scenario-name 1_all_nodes

The flat fixture layout avoids using a directory named "output/" (which is
gitignored) and stores only the files needed by the validation pipeline.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml

from _common import repo_root


FIXTURE_ROOT = repo_root() / "fixtures" / "cluster_results"


def _read_tsv(experiment_dir: Path) -> dict[str, tuple[str, str]]:
    """Parse sequential_jobs.tsv -> {scenario_name: (run_job_id, link_bench_job_id)}."""
    tsv_path = experiment_dir / "sequential_jobs.tsv"
    if not tsv_path.exists():
        raise SystemExit(f"sequential_jobs.tsv not found in {experiment_dir}")
    result: dict[str, tuple[str, str]] = {}
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            result[row["scenario"]] = (row["run_job_id"], row["link_bench_job_id"])
    return result


def _derive_fixture_id(scenario_dir: Path, run_job_id: str) -> str:
    """Derive a fixture ID from the experiment/scenario/job structure.

    Examples:
        experiments/experiment_1/1_all_nodes, job 15 -> exp1_1_all_nodes_run15
        experiments/experiment_2/9_g_only_pp4, job 65 -> exp2_9_g_only_pp4_run65
    """
    scenario_name = scenario_dir.name
    experiment_dir = scenario_dir.parent
    experiment_name = experiment_dir.name
    short = experiment_name.replace("experiment_", "exp")
    return f"{short}_{scenario_name}_run{run_job_id}"


def _validate_source(
    scenario_dir: Path,
    run_job_id: str,
    link_bench_job_id: str,
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate all required source files exist. Returns key paths."""
    base_config = scenario_dir / "hops.base.yaml"
    if not base_config.exists():
        raise SystemExit(f"hops.base.yaml not found: {base_config}")

    run_dir = scenario_dir / "output" / run_job_id
    if not run_dir.is_dir():
        raise SystemExit(f"run output dir not found: {run_dir}")

    trace_map = run_dir / "hops_trace_map.json"
    if not trace_map.exists():
        raise SystemExit(f"hops_trace_map.json not found: {trace_map}")

    megatron_summary = run_dir / "megatron_summary.json"
    if not megatron_summary.exists():
        raise SystemExit(f"megatron_summary.json not found: {megatron_summary}")

    trace_dir = run_dir / "megatron_trace"
    jsonl_files = sorted(trace_dir.glob("*.jsonl")) if trace_dir.is_dir() else []
    if not jsonl_files:
        raise SystemExit(f"no megatron_trace/*.jsonl in {run_dir}")

    lb_dir = scenario_dir / "output" / link_bench_job_id / "calibration" / "link_bench"
    lb_files = sorted(lb_dir.glob("*.jsonl")) if lb_dir.is_dir() else []
    if not lb_files:
        raise SystemExit(
            f"no link_bench/*.jsonl in {lb_dir}\n"
            f"  (link_bench_job_id={link_bench_job_id})"
        )

    return base_config, run_dir, trace_map, megatron_summary, lb_dir


def extract(
    scenario_dir: Path,
    run_job_id: str,
    link_bench_job_id: str,
    fixture_id: str | None = None,
) -> Path:
    """Extract a fixture from experiment output into fixtures/cluster_results/."""
    base_config, run_dir, trace_map, megatron_summary, lb_dir = _validate_source(
        scenario_dir, run_job_id, link_bench_job_id
    )

    if fixture_id is None:
        fixture_id = _derive_fixture_id(scenario_dir, run_job_id)

    dest = FIXTURE_ROOT / fixture_id
    if dest.exists():
        raise SystemExit(
            f"fixture already exists: {dest}\n"
            f"  Remove it first if you want to re-extract."
        )

    dest.mkdir(parents=True)

    shutil.copy2(base_config, dest / "hops.base.yaml")
    shutil.copy2(trace_map, dest / "hops_trace_map.json")
    shutil.copy2(megatron_summary, dest / "megatron_summary.json")
    shutil.copytree(run_dir / "megatron_trace", dest / "megatron_trace")
    shutil.copytree(lb_dir, dest / "calibration" / "link_bench")

    manifest = {
        "source_scenario": str(scenario_dir.relative_to(repo_root())),
        "job_id": run_job_id,
        "link_bench_job_id": link_bench_job_id,
        "link_bench_source": str(lb_dir.relative_to(repo_root())),
        "description": f"Extracted from {scenario_dir.name} run {run_job_id}",
    }
    (dest / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    size_kb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1024
    print(f"extracted fixture: {fixture_id}")
    print(f"  path: {dest}")
    print(f"  size: {size_kb:.0f} KB")
    return dest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_argument_group("explicit mode")
    group.add_argument("--scenario", help="Path to scenario dir (e.g. experiments/experiment_1/1_all_nodes)")
    group.add_argument("--run-job-id", help="Slurm job ID for the training run")
    group.add_argument("--link-bench-job-id", help="Slurm job ID for the link bench run")

    tsv_group = parser.add_argument_group("TSV mode (reads sequential_jobs.tsv)")
    tsv_group.add_argument("--experiment", help="Path to experiment dir (e.g. experiments/experiment_2)")
    tsv_group.add_argument("--scenario-name", help="Scenario name from sequential_jobs.tsv")

    parser.add_argument("--all", action="store_true", help="Extract all scenarios from --experiment")
    parser.add_argument("--fixture-id", default=None, help="Override the auto-derived fixture ID")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.experiment and args.all:
        experiment_dir = Path(args.experiment).resolve()
        tsv = _read_tsv(experiment_dir)
        for scenario_name, (run_job_id, lb_job_id) in tsv.items():
            scenario_dir = experiment_dir / scenario_name
            fixture_id = _derive_fixture_id(scenario_dir, run_job_id)
            if (FIXTURE_ROOT / fixture_id).exists():
                print(f"skipping {fixture_id} (already exists)")
                continue
            extract(scenario_dir, run_job_id, lb_job_id)
        return

    if args.experiment and args.scenario_name:
        experiment_dir = Path(args.experiment).resolve()
        tsv = _read_tsv(experiment_dir)
        if args.scenario_name not in tsv:
            available = ", ".join(sorted(tsv))
            raise SystemExit(
                f"scenario '{args.scenario_name}' not in sequential_jobs.tsv\n"
                f"  available: {available}"
            )
        run_job_id, lb_job_id = tsv[args.scenario_name]
        scenario_dir = experiment_dir / args.scenario_name
    elif args.scenario and args.run_job_id and args.link_bench_job_id:
        scenario_dir = Path(args.scenario).resolve()
        run_job_id = args.run_job_id
        lb_job_id = args.link_bench_job_id
    else:
        raise SystemExit(
            "Provide either:\n"
            "  --scenario + --run-job-id + --link-bench-job-id\n"
            "or:\n"
            "  --experiment + --scenario-name\n"
            "or:\n"
            "  --experiment + --all"
        )

    extract(scenario_dir, run_job_id, lb_job_id, fixture_id=args.fixture_id)


if __name__ == "__main__":
    main()
