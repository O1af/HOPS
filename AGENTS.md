# HOPS AutoResearch Playbook

Improve HOPS prediction accuracy against real Megatron-LM traces. Current
no_lookahead MAPE is ~92% (link_calibrated: ~12%). Reduce the analytical
model's error toward single digits while preserving rank-order fidelity.
Work only on the **train split**. The test set is invisible to you.

## Setup

```bash
uv run python experiments/tools/split_fixtures.py --apply      # one-time: create train/test splits
uv run pytest tests/                                           # unit tests pass
uv run python experiments/tools/validate_fixtures.py --split train  # full train validation
```

## Test Set Isolation

- **NEVER** read files under `fixtures/test/`, run `--split test`, or use test fixture error patterns to inform hypotheses.
- Diagnostic fixtures (`fixtures/diagnostic/`) are for sanity checks only, not parameter fitting.
- Test evaluation is **human-initiated only**.
- **Why:** The test set measures generalization to unseen configs. Observing its errors contaminates the measurement.

## Human Test-Bench Protocol

The human (not the agent) runs the test split at two checkpoints:

1. **Before** the autoresearch loop — baseline generalization score
2. **After** the loop completes — measure whether improvements generalized

```bash
uv run python experiments/tools/validate_fixtures.py --split test
```

If train MAPE dropped but test MAPE didn't, the agent overfit to experiment_2 quirks.

## The Research Loop

### OBSERVE — Analyze training errors

```bash
uv run python experiments/tools/validate_fixtures.py --split train              # overview + group breakdown
uv run python experiments/tools/validate_fixtures.py --split train --verbose    # phase breakdowns
uv run python experiments/tools/validate_fixtures.py --split train --fixture <name> --verbose
uv run python experiments/tools/inspect_fixture.py --fixture <name>             # raw trace ground truth
uv run python experiments/tools/inspect_fixture.py --fixture <name> --phase FORWARD  # filter by phase
```

- **Inspect ground truth first.** Use `inspect_fixture.py` to see actual per-stage compute timings, per-link transfer durations, and iteration wall times from the real Megatron trace. Compare these against what the sim predicts to pinpoint where the model is wrong.
- Group errors by device type, scheduler, pipeline depth, batch size.
- **Variant gap analysis:**
  - `no_lookahead → link_calibrated` gap = transfer/framework modeling error.
  - `link_calibrated → trace_replay` gap = compute estimation error.
  - If trace_replay barely helps, pipeline orchestration (scheduling, contention) is the problem.
- Phase breakdown shows real vs sim per (stage, phase). Largest ms delta = top error contributor.
- Spearman rho < 0.5 means rank-order is broken — fix that before chasing absolute error.

### HYPOTHESIZE — Falsifiable prediction

Good: *"A10G forward compute is 3x overestimated because efficiency 0.45 is too low for this memory-bound shape. Raising it to 0.7 should drop exp2_13 from +203% to within +30%."*

Bad: *"Try lowering all efficiencies."*

| Error category | Symptom | Change site |
|---|---|---|
| Compute efficiency per device | Systematic error grouped by device | `presets.py`, `config.py` DEFAULT_COMPUTE_EFFICIENCY |
| Backward factor | bubble_pp_delta consistently one-sided | `compute_model.py`, config backward_factor |
| Transfer model | comm_overhead_delta large | `core/timing.py`, `hardware/topology.py` |
| Launch overhead | Small-model fixtures disproportionately wrong | `presets.py` launch_overhead_ms |
| Framework overhead | Sim faster even with correct compute | iteration_barrier in `derive_megatron_stats.py` |
| Scheduler modeling | Differential error between GPipe/1F1B/ZeroBubble | `core/scheduler.py` |
| Memory bandwidth | Memory-bound stages (large activations) worse | `presets.py` memory_bandwidth_gbps |

### IMPLEMENT — One targeted change

- Directly tests one hypothesis. Prefer constant fixes over logic changes.
- Never change multiple independent parameters simultaneously.

### VALIDATE

```bash
uv run pytest tests/                                                             # 1. unit tests
uv run python experiments/tools/validate_fixtures.py --split train --fixture <X>  # 2. target fixture
uv run python experiments/tools/validate_fixtures.py --split train                # 3. no regressions
uv run python experiments/tools/validate_fixtures.py --split train --update-golden # 4. lock in
```

Before updating golden: MAPE down or held, MAE down or held, Spearman up or held, no fixture regressed beyond tolerance, change is physically grounded. **Git commit after each golden update** describing hypothesis and result.

### REFLECT

Record hypothesis, change, before/after metrics. If MAPE dropped significantly, restart from OBSERVE — the error landscape shifted.

## Anti-Patterns

- **Overfitting to noise:** exp2 has duplicate runs for some configs. If improvement only helps one of a pair, you're fitting run variance.
- **Global constants for local problems:** If only A10G fixtures are wrong, don't change DEFAULT_COMPUTE_EFFICIENCY globally.
- **Ignoring physics:** Roofline (`max(compute_time, memory_time)`) is physically grounded. No negative efficiencies or bandwidth above spec sheet.
- **Fixing symptoms not causes:** Wrong bubble_pp_delta usually stems from wrong compute timing. Fix compute first.
- **Too many knobs at once:** One conceptual change per iteration or you can't attribute results.

## Starting Hypotheses

1. **Device efficiency mismatch.** Default 0.45 calibrated on H100. A10G/L4 may differ due to different tensor cores and memory systems. Check if A10G/L4 fixture errors are systematically larger.
2. **Backward factor.** Default 2.0 may not match reality. Compare per-stage backward timing in `--verbose` against the 2x assumption.
3. **GPipe amplification.** GPipe's fill/drain bubbles amplify compute errors more than 1F1B. If GPipe fixtures are disproportionately wrong, the root cause is compute, not scheduling.
4. **Memory bandwidth bottleneck on A10G/L4.** A10G: 600 GB/s, L4: 300 GB/s vs H100: 3350 GB/s. If workloads are memory-bound on these devices but sim treats them as compute-bound, latency will be massively under-predicted.

## Tooling Review

**Strengths:** Single-fixture validation in seconds. Full train validation ~2 min (8 parallel workers). Automatic regression detection (exit 0/1). Three-variant attribution (analytical → link calibrated → trace replay). Per-operation phase breakdown (PR #14). Dynamic transfer contention (PR #15). Per-device group breakdown in suite summary (h100, a10g, l4, g_family, mixed). Split-safe golden merge (`--update-golden` preserves fixture entries from other splits).

**Gaps:** No ablation scripting — parameter sweeps require manual YAML editing. No improvement history — golden overwrites per-fixture entries silently (workaround: git commits). No scheduler-level or PP-depth grouping (only device grouping is automated).

## Architecture Reference

```
src/hops/
  config.py               DEFAULT_COMPUTE_EFFICIENCY=0.45, DEFAULT_MEMORY_EFFICIENCY=0.6
  presets.py              h100: 989T, a10g: 125T, l4: 121T, l40s: 362T
  latency/compute_model.py  roofline: max(compute_time, memory_time)
  core/timing.py          transfer scheduling, bandwidth contention
  core/pipeline.py        event orchestration, iteration_barrier
  core/scheduler.py       GPipe, 1F1B, ZeroBubble policies
  hardware/topology.py    device graph, locality penalties

experiments/tools/
  validate_fixtures.py    validation entry point (--split train, --fixture X, --update-golden)
  inspect_fixture.py      raw trace inspection (per-stage timing, per-link, iteration stats)
  compare_run.py          per-fixture comparison + phase breakdown
  split_fixtures.py       apply train/test/diagnostic/archive splits

fixtures/
  expected_metrics.json   golden baseline
  train/cluster_results/  training fixtures (experiment_2)
  test/cluster_results/   OFF LIMITS (experiment_3, scenarios 1-42)
  diagnostic/             sanity checks only (experiment_3, scenarios 43-48)
```
