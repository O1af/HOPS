"""Materialize a flat fixture into the scenario/output/<job_id>/ layout
expected by run_validation.run().

The fixture tree uses a flat layout (no directory named "output/") so that
.gitignore does not exclude it. This loader reconstitutes the standard
layout in a working directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


SPLITS = ("train", "test", "diagnostic", "archive")

_FIXTURES_ROOT = Path(__file__).resolve().parent


def materialize_fixture(fixture_dir: Path, workdir: Path) -> tuple[Path, str]:
    """Copy fixture data into the scenario layout that run_validation expects.

    Returns (scenario_dir, job_id) suitable for::

        run_validation.run(scenario_dir, job_id)
    """
    manifest_path = fixture_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    job_id = str(manifest["job_id"])

    shutil.copy2(fixture_dir / "hops.base.yaml", workdir / "hops.base.yaml")

    job_dir = workdir / "output" / job_id
    job_dir.mkdir(parents=True)

    shutil.copy2(fixture_dir / "hops_trace_map.json", job_dir / "hops_trace_map.json")
    shutil.copy2(fixture_dir / "megatron_summary.json", job_dir / "megatron_summary.json")

    shutil.copytree(fixture_dir / "megatron_trace", job_dir / "megatron_trace")

    calibration_src = fixture_dir / "calibration"
    if calibration_src.is_dir():
        shutil.copytree(calibration_src, job_dir / "calibration")

    return workdir, job_id


def discover_fixtures(
    fixtures_root: Path | None = None,
    split: str = "all",
    *,
    fixtures_base: Path | None = None,
) -> list[Path]:
    """Return sorted list of fixture directories containing manifest.yaml.

    When *split* is ``"all"``, searches every managed split subdirectory
    (train/, test/, diagnostic/, archive/) plus the legacy flat layout
    (cluster_results/).  When *split* names a specific split, only that
    subdirectory is searched.

    Falls back to the flat ``cluster_results/`` layout when no split
    subdirectories have been created yet.
    """
    root = fixtures_base or fixtures_root or _FIXTURES_ROOT

    if split != "all":
        split_root = root / split / "cluster_results"
        if split_root.is_dir():
            return sorted(p.parent for p in split_root.glob("*/manifest.yaml"))
        return []

    seen: set[Path] = set()
    results: list[Path] = []

    for s in SPLITS:
        split_root = root / s / "cluster_results"
        if split_root.is_dir():
            for p in split_root.glob("*/manifest.yaml"):
                if p.parent not in seen:
                    seen.add(p.parent)
                    results.append(p.parent)

    flat_root = root / "cluster_results"
    if flat_root.is_dir():
        for p in flat_root.glob("*/manifest.yaml"):
            if p.parent not in seen:
                seen.add(p.parent)
                results.append(p.parent)

    return sorted(results, key=lambda p: p.name)
