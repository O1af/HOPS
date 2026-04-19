# AutoResearch Agent Instructions for HOPS

## Validation Command

After any change to the simulator, run:

```
uv run python experiments/tools/validate_fixtures.py
```

This runs the full validation pipeline against real cluster data and reports
whether accuracy regressed. Exit code 0 means no regression.

## Quick Feedback (single fixture)

```
uv run python experiments/tools/validate_fixtures.py --fixture exp1_1_all_nodes_run15
```

## Extracting New Fixtures

From an experiment with explicit job IDs:

```
uv run python experiments/tools/extract_fixture.py \
    --scenario experiments/experiment_1/1_all_nodes \
    --run-job-id 15 --link-bench-job-id 16
```

From an experiment with sequential_jobs.tsv:

```
uv run python experiments/tools/extract_fixture.py \
    --experiment experiments/experiment_2 \
    --scenario-name 1_all_nodes
```

## Updating the Baseline

After a confirmed improvement, lock in the new baseline:

```
uv run python experiments/tools/validate_fixtures.py --update-golden
```

## Running Tests

```
uv run pytest                    # fast unit tests only
uv run pytest -m slow            # validation regression tests
```

## Key Metrics

All metrics matter. Researchers use HOPS to model novel scheduling and training
algorithms, so the sim must be accurate enough that relative comparisons between
configurations are trustworthy — if the sim says config A has less bubble time
than config B, that should hold in reality.

- **throughput_error_pct** — throughput prediction vs real Megatron traces. Lower is better. Current baseline: ~139% mean error.
- **bubble_pp_delta** — pipeline bubble ratio accuracy (percentage points). Critical for scheduling algorithm design.
- **util_spearman_rho** — rank correlation of per-stage utilization. Critical for load balancing decisions. Higher is better (1.0 = perfect).
- **comm_overhead_delta** — communication overhead prediction error.
- **latency** — end-to-end iteration latency accuracy.

Rank-order fidelity matters as much as absolute error.

## Architecture

- `src/hops/` -- the simulator package
- `experiments/tools/` -- validation, comparison, and fixture toolchain
- `fixtures/cluster_results/` -- real cluster data fixtures (flat layout)
- `fixtures/loader.py` -- reconstitutes fixtures into run_validation layout
- `fixtures/expected_metrics.json` -- golden baseline for regression detection
- `main.py` -- HOPS CLI entry point

