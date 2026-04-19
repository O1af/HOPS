"""Regression tests that run the full validation pipeline against fixtures.

Marked slow — run with: uv run pytest -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "experiments" / "tools"
_FIXTURES_DIR = _REPO_ROOT / "fixtures"

if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
if str(_FIXTURES_DIR) not in sys.path:
    sys.path.insert(0, str(_FIXTURES_DIR))

import aggregate  # noqa: E402
import run_validation  # noqa: E402
from loader import discover_fixtures, materialize_fixture  # noqa: E402
from validate_fixtures import TOLERANCES, _load_golden  # noqa: E402

_FIXTURE_MAP = {f.name: f for f in discover_fixtures()}
_GOLDEN = _load_golden()


@pytest.mark.slow
@pytest.mark.parametrize("fixture_name", sorted(_FIXTURE_MAP))
def test_fixture_does_not_regress(fixture_name: str, tmp_path: Path) -> None:
    fixture_dir = _FIXTURE_MAP[fixture_name]

    scenario_dir, job_id = materialize_fixture(fixture_dir, tmp_path)
    comparison_path = run_validation.run(scenario_dir, job_id)
    result = aggregate.score_comparison(fixture_name, comparison_path)

    tols = _GOLDEN.get("tolerances", TOLERANCES)
    golden_fixture = _GOLDEN.get("fixtures", {}).get(fixture_name, {})

    for name in aggregate.SCORED_VARIANTS:
        vs = result.variant_scores.get(name)
        if vs is None:
            continue
        gv = golden_fixture.get(name, {})

        if vs.throughput_error_pct is not None:
            g = gv.get("throughput_error_pct")
            if g is not None:
                delta = abs(vs.throughput_error_pct) - abs(g)
                assert delta <= tols.get("throughput_error_pct", 2.0), (
                    f"{fixture_name}/{name} throughput: |err| {abs(vs.throughput_error_pct):.2f}% "
                    f"(golden {abs(g):.2f}%, delta +{delta:.2f}pp)"
                )

        if vs.bubble_pp_delta is not None:
            g = gv.get("bubble_pp_delta")
            if g is not None:
                delta = abs(vs.bubble_pp_delta) - abs(g)
                assert delta <= tols.get("bubble_pp_delta", 2.0), (
                    f"{fixture_name}/{name} bubble: |delta| {abs(vs.bubble_pp_delta):.2f}pp "
                    f"(golden {abs(g):.2f}pp, delta +{delta:.2f}pp)"
                )

        if vs.util_spearman_rho is not None:
            g = gv.get("util_spearman_rho")
            if g is not None:
                drop = g - vs.util_spearman_rho
                assert drop <= tols.get("util_spearman_rho", 0.1), (
                    f"{fixture_name}/{name} spearman: {vs.util_spearman_rho:.3f} "
                    f"(golden {g:.3f}, dropped {drop:.3f})"
                )
