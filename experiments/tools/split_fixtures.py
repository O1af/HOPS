"""Move flat validation fixtures into train/test/diagnostic/archive splits.

The migration is intentionally conservative: it computes all destination paths
first and fails before moving anything if a destination already exists.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from _common import repo_root


SPLITS = ("train", "test", "diagnostic", "archive")


@dataclass(frozen=True)
class PlannedMove:
    source: Path
    split: str
    fixture_id: str

    @property
    def destination(self) -> Path:
        return repo_root() / "fixtures" / self.split / "cluster_results" / self.fixture_id


def _read_tsv(experiment_dir: Path) -> dict[str, tuple[str, str]]:
    tsv_path = experiment_dir / "sequential_jobs.tsv"
    if not tsv_path.exists():
        raise SystemExit(f"sequential_jobs.tsv not found: {tsv_path}")

    rows: dict[str, tuple[str, str]] = {}
    with tsv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows[row["scenario"]] = (row["run_job_id"], row["link_bench_job_id"])
    return rows


def _fixture_id(experiment_dir: Path, scenario_name: str, run_job_id: str) -> str:
    short = experiment_dir.name.replace("experiment_", "exp")
    return f"{short}_{scenario_name}_run{run_job_id}"


def expected_split_ids() -> dict[str, set[str]]:
    root = repo_root()
    exp2_dir = root / "experiments" / "experiment_2"
    exp3_dir = root / "experiments" / "experiment_3"

    train = {
        _fixture_id(exp2_dir, scenario_name, run_job_id)
        for scenario_name, (run_job_id, _lb_job_id) in _read_tsv(exp2_dir).items()
    }

    test: set[str] = set()
    diagnostic: set[str] = set()
    for scenario_name, (run_job_id, _lb_job_id) in _read_tsv(exp3_dir).items():
        fixture_id = _fixture_id(exp3_dir, scenario_name, run_job_id)
        prefix = scenario_name.split("_", 1)[0]
        if prefix.isdigit() and 1 <= int(prefix) <= 42:
            test.add(fixture_id)
        elif scenario_name.startswith(("43_diag", "44_diag", "45_diag", "46_diag", "47_diag", "48_diag")):
            diagnostic.add(fixture_id)
        else:
            raise SystemExit(f"could not classify experiment_3 scenario: {scenario_name}")

    return {
        "train": train,
        "test": test,
        "diagnostic": diagnostic,
    }


def plan_moves() -> list[PlannedMove]:
    flat_root = repo_root() / "fixtures" / "cluster_results"
    if not flat_root.exists():
        raise SystemExit(f"legacy fixture root not found: {flat_root}")

    expected = expected_split_ids()
    moves: list[PlannedMove] = []
    for fixture_dir in sorted(flat_root.iterdir(), key=lambda p: p.name):
        if not (fixture_dir / "manifest.yaml").exists():
            continue

        split = "archive"
        for candidate in ("train", "test", "diagnostic"):
            if fixture_dir.name in expected[candidate]:
                split = candidate
                break

        moves.append(
            PlannedMove(
                source=fixture_dir,
                split=split,
                fixture_id=fixture_dir.name,
            )
        )
    return moves


def _print_plan(moves: list[PlannedMove]) -> None:
    counts = {split: 0 for split in SPLITS}
    for move in moves:
        counts[move.split] += 1

    print("planned fixture split counts:")
    for split in SPLITS:
        print(f"  {split}: {counts[split]}")

    print()
    for move in moves:
        rel_src = move.source.relative_to(repo_root())
        rel_dst = move.destination.relative_to(repo_root())
        print(f"{move.split:10s} {rel_src} -> {rel_dst}")


def _validate_destinations(moves: list[PlannedMove]) -> None:
    conflicts = [move.destination for move in moves if move.destination.exists()]
    if conflicts:
        message = "\n".join(f"  {path}" for path in conflicts[:20])
        extra = "" if len(conflicts) <= 20 else f"\n  ... and {len(conflicts) - 20} more"
        raise SystemExit(f"destination fixture(s) already exist:\n{message}{extra}")


def apply_moves(moves: list[PlannedMove]) -> None:
    _validate_destinations(moves)

    for move in moves:
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(move.source), str(move.destination))

    flat_root = repo_root() / "fixtures" / "cluster_results"
    if flat_root.exists() and not any(flat_root.iterdir()):
        flat_root.rmdir()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print planned moves without changing files")
    mode.add_argument("--apply", action="store_true", help="Move fixtures into split roots")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    moves = plan_moves()
    _print_plan(moves)

    if args.apply:
        print()
        apply_moves(moves)
        print("fixture split applied")


if __name__ == "__main__":
    main()
