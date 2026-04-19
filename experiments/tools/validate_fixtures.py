"""Run the HOPS validation pipeline against all checked-in fixtures.

Single command for agent iteration:

    uv run python experiments/tools/validate_fixtures.py

Exit 0 = no regression.  Exit 1 = regression detected.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path

from _common import repo_root

sys.path.insert(0, str(repo_root() / "fixtures"))

import aggregate  # noqa: E402
import run_validation  # noqa: E402
from loader import discover_fixtures, materialize_fixture  # noqa: E402


GOLDEN_PATH = repo_root() / "fixtures" / "expected_metrics.json"

TOLERANCES = {
    "throughput_error_pct": 2.0,
    "bubble_pp_delta": 2.0,
    "util_spearman_rho": 0.1,
}


def _load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _save_golden(suite: aggregate.SuiteResult) -> None:
    doc: dict = {
        "schema_version": 2,
        "tolerances": TOLERANCES,
        "fixtures": {},
        "suite": {
            "throughput_mape": round(suite.aggregates.throughput_mape, 2),
            "bubble_mae_pp": round(suite.aggregates.bubble_mae_pp, 2),
            "util_spearman_mean": round(suite.aggregates.util_spearman_mean, 3) if suite.aggregates.util_spearman_mean is not None else None,
        },
    }
    for fr in suite.fixtures:
        per_variant: dict = {}
        for name, vs in fr.variant_scores.items():
            entry: dict = {}
            if vs.throughput_error_pct is not None:
                entry["throughput_error_pct"] = round(vs.throughput_error_pct, 2)
            if vs.bubble_pp_delta is not None:
                entry["bubble_pp_delta"] = round(vs.bubble_pp_delta, 2)
            if vs.util_spearman_rho is not None:
                entry["util_spearman_rho"] = round(vs.util_spearman_rho, 3)
            if entry:
                per_variant[name] = entry
        doc["fixtures"][fr.fixture_id] = per_variant

    GOLDEN_PATH.write_text(
        json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"updated golden: {GOLDEN_PATH}")


@contextlib.contextmanager
def _suppress_all_output():
    """Suppress both Python-level and subprocess stdout/stderr."""
    import os
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull)


def _run_fixture(fixture_dir: Path, verbose: bool = False) -> aggregate.FixtureResult:
    fixture_id = fixture_dir.name
    with tempfile.TemporaryDirectory(prefix=f"hops_val_{fixture_id}_") as tmpdir:
        workdir = Path(tmpdir)
        scenario_dir, job_id = materialize_fixture(fixture_dir, workdir)
        if verbose:
            comparison_path = run_validation.run(scenario_dir, job_id)
        else:
            with _suppress_all_output():
                comparison_path = run_validation.run(scenario_dir, job_id)
        return aggregate.score_comparison(fixture_id, comparison_path)


def _fmt(val: float | None, spec: str = "+.1f", suffix: str = "") -> str:
    if val is None:
        return "--"
    return f"{val:{spec}}{suffix}"


def _check_regression(
    suite: aggregate.SuiteResult,
    golden: dict,
) -> list[str]:
    """Return list of regression messages. Empty = pass."""
    tols = golden.get("tolerances", TOLERANCES)
    golden_fixtures = golden.get("fixtures", {})
    issues: list[str] = []

    for fr in suite.fixtures:
        gf = golden_fixtures.get(fr.fixture_id, {})
        for name, vs in fr.variant_scores.items():
            gv = gf.get(name, {})

            if vs.throughput_error_pct is not None:
                g = gv.get("throughput_error_pct")
                if g is not None:
                    delta = abs(vs.throughput_error_pct) - abs(g)
                    if delta > tols.get("throughput_error_pct", 2.0):
                        issues.append(
                            f"  {fr.fixture_id}/{name} throughput: "
                            f"|err| {abs(vs.throughput_error_pct):.1f}% "
                            f"(was {abs(g):.1f}%, +{delta:.1f}pp)"
                        )

            if vs.bubble_pp_delta is not None:
                g = gv.get("bubble_pp_delta")
                if g is not None:
                    delta = abs(vs.bubble_pp_delta) - abs(g)
                    if delta > tols.get("bubble_pp_delta", 2.0):
                        issues.append(
                            f"  {fr.fixture_id}/{name} bubble: "
                            f"|delta| {abs(vs.bubble_pp_delta):.1f}pp "
                            f"(was {abs(g):.1f}pp, +{delta:.1f}pp)"
                        )

            if vs.util_spearman_rho is not None:
                g = gv.get("util_spearman_rho")
                if g is not None:
                    drop = g - vs.util_spearman_rho
                    if drop > tols.get("util_spearman_rho", 0.1):
                        issues.append(
                            f"  {fr.fixture_id}/{name} spearman: "
                            f"{vs.util_spearman_rho:.3f} "
                            f"(was {g:.3f}, dropped {drop:.3f})"
                        )

    return issues


def _print_summary(suite: aggregate.SuiteResult, golden: dict) -> None:
    golden_fixtures = golden.get("fixtures", {})

    print()
    print("=" * 120)
    print("FIXTURE DETAILS")
    print("=" * 120)

    for fr in suite.fixtures:
        m = fr.megatron
        gf = golden_fixtures.get(fr.fixture_id, {})
        print()
        print(f">>> {fr.fixture_id}")
        mt_parts = []
        if m.throughput_per_s is not None:
            mt_parts.append(f"tput={m.throughput_per_s:.1f} sam/s")
        if m.latency_mean_ms is not None:
            mt_parts.append(f"lat={m.latency_mean_ms:.0f}ms")
        if m.bubble_ratio is not None:
            mt_parts.append(f"bubble={m.bubble_ratio:.1f}%")
        if m.completed_microbatches is not None:
            mt_parts.append(f"ub={m.completed_microbatches}")
        print(f"    megatron: {' | '.join(mt_parts)}")

        hdr = (
            f"    {'variant':<18s} "
            f"{'tput':>8s} {'tput_err':>9s} "
            f"{'lat_ms':>8s} "
            f"{'bubble':>8s} {'bub_Δpp':>8s} "
            f"{'comm%':>7s} {'commΔ':>7s} "
            f"{'ρ_spear':>8s} {'u_maxΔ':>7s}"
        )
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))

        for vname in aggregate.SCORED_VARIANTS:
            vs = fr.variant_scores.get(vname)
            if vs is None:
                print(f"    {vname:<18s} {'--':>8s}")
                continue
            print(
                f"    {vname:<18s} "
                f"{_fmt(vs.throughput_per_s, '.1f'):>8s} "
                f"{_fmt(vs.throughput_error_pct, '+.1f', '%'):>9s} "
                f"{_fmt(vs.latency_mean_ms, '.0f'):>8s} "
                f"{_fmt(vs.bubble_ratio, '.1f', '%'):>8s} "
                f"{_fmt(vs.bubble_pp_delta, '+.1f'):>8s} "
                f"{_fmt(vs.comm_overhead_ratio, '.1f', '%'):>7s} "
                f"{_fmt(vs.comm_overhead_delta, '+.1f'):>7s} "
                f"{_fmt(vs.util_spearman_rho, '.3f'):>8s} "
                f"{_fmt(vs.util_max_abs_delta, '.1f', '%'):>7s}"
            )

        for vname in aggregate.SCORED_VARIANTS:
            vs = fr.variant_scores.get(vname)
            if vs is None or vs.per_stage_util is None:
                continue
            stages = vs.per_stage_util
            parts = [f"s{s['stage']}:{s['hops']:.0f}/{s['megatron']:.0f}" for s in stages]
            print(f"    {vname} util(hops/meg): {' '.join(parts)}")

    agg = suite.aggregates
    golden_suite = golden.get("suite", {})

    print()
    print("=" * 120)
    print("SUITE AGGREGATES")
    print("=" * 120)

    def _agg_line(label: str, val: float | None, golden_key: str, fmt: str, suffix: str) -> None:
        g = golden_suite.get(golden_key)
        val_str = f"{val:{fmt}}{suffix}" if val is not None else "--"
        g_str = f" (golden: {g:{fmt}}{suffix})" if g is not None else ""
        print(f"  {label:<24s} {val_str}{g_str}")

    _agg_line("throughput MAPE", agg.throughput_mape, "throughput_mape", ".1f", "%")
    _agg_line("bubble MAE", agg.bubble_mae_pp, "bubble_mae_pp", ".1f", "pp")
    _agg_line("util Spearman mean", agg.util_spearman_mean, "util_spearman_mean", ".3f", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=None, help="Run only this fixture (by directory name)")
    parser.add_argument("--update-golden", action="store_true", help="Overwrite expected_metrics.json")
    parser.add_argument("--verbose", action="store_true", help="Show full HOPS output per fixture")
    args = parser.parse_args()

    all_fixtures = discover_fixtures()
    if not all_fixtures:
        raise SystemExit("no fixtures found in fixtures/cluster_results/")

    if args.fixture:
        all_fixtures = [f for f in all_fixtures if f.name == args.fixture]
        if not all_fixtures:
            raise SystemExit(f"fixture not found: {args.fixture}")

    n = len(all_fixtures)
    results: list[aggregate.FixtureResult] = []
    for i, fixture_dir in enumerate(all_fixtures, 1):
        sys.stdout.write(f"[{i}/{n}] {fixture_dir.name} ... ")
        sys.stdout.flush()
        result = _run_fixture(fixture_dir, verbose=args.verbose)
        nl = result.variant_scores.get("no_lookahead")
        lc = result.variant_scores.get("link_calibrated")
        nl_str = _fmt(nl.throughput_error_pct if nl else None, "+.1f", "%")
        lc_str = _fmt(lc.throughput_error_pct if lc else None, "+.1f", "%")
        print(f"nl={nl_str}  lc={lc_str}")
        results.append(result)

    suite = aggregate.compute_suite_score(results)
    golden = _load_golden()

    _print_summary(suite, golden)

    if args.update_golden:
        print()
        _save_golden(suite)
        print("RESULT: golden updated")
        return

    issues = _check_regression(suite, golden)
    if issues:
        print()
        print("REGRESSION DETECTED:")
        for issue in issues:
            print(issue)
        sys.exit(1)
    else:
        print()
        print("PASS (no regression)")


if __name__ == "__main__":
    main()
