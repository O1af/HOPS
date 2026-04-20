# HOPS AutoResearch Playbook

Improve HOPS prediction accuracy against real Megatron-LM traces by fixing
the simulator's modeling logic — not just tuning constants. Current
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

## Research Log — `AutoResearch.md`

Every iteration (landed or reverted) is recorded in [`AutoResearch.md`](AutoResearch.md).

- **Read it before OBSERVE.** The last entry tells you the current error
  landscape, which hypotheses have already been tried, and which were
  reverted and why. Do not repeat reverted experiments without new evidence.
- **Append one entry per iteration**, using the template at the top of the
  file. Record reverted attempts too — dead ends save the next agent time.
- **Use the numbers from `validate_fixtures.py --split train`** (SUITE
  AGGREGATES and GROUP BREAKDOWN). Don't paraphrase.
- **Link the commit** when a change lands and golden is updated. The golden
  file is gitignored locally, so the commit is the durable record.

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

Good: *"A10G forward compute is 3x overestimated because the roofline ignores memory bandwidth on low-BW devices. Adding a `max(compute_time, mem_bw_time)` path in `compute_model.py` should drop exp2_13 from +203% to within +30%."*

Bad: *"Try lowering all efficiencies."* (tuning a constant without fixing the model)

| Error category | Symptom | Fix (model logic, not just constants) |
|---|---|---|
| Compute model missing physics | Systematic error grouped by device or shape | Add memory-BW path to roofline in `compute_model.py`; add device-specific scaling in `presets.py` |
| Backward scaling | bubble_pp_delta consistently one-sided | Fix backward-time derivation in `compute_model.py` (e.g., layer-type-dependent ratio) |
| Transfer model | comm_overhead_delta large | Fix contention, overlap, or topology modeling in `core/timing.py`, `hardware/topology.py` |
| Launch / framework overhead | Small-model fixtures disproportionately wrong | Model kernel-launch amortization or framework sync in `core/pipeline.py` |
| Scheduler modeling | Differential error between GPipe/1F1B/ZeroBubble | Fix fill/drain logic or micro-batch pipelining in `core/scheduler.py` |
| Missing interaction effects | Error grows with pipeline depth or batch size | Add cross-stage contention, memory pressure, or GC modeling |

### IMPLEMENT — One targeted change

- Fix the simulator's modeling logic in `src/hops/`. Constant tweaks (efficiencies,
  overheads) are low-value — they mask wrong physics. Prefer adding or correcting
  a modeling equation, scheduling rule, or bandwidth formula over tuning a number.
  Examples: add a memory-bandwidth path to the roofline, fix how backward compute
  scales, model cross-node contention that was previously ignored.
- A constant change is acceptable only when the current value is demonstrably
  wrong for a specific device (e.g., spec-sheet bandwidth was entered incorrectly).
- Never change multiple independent things simultaneously.

### VALIDATE

```bash
uv run pytest tests/                                                             # 1. unit tests
uv run python experiments/tools/validate_fixtures.py --split train --fixture <X>  # 2. target fixture
uv run python experiments/tools/validate_fixtures.py --split train                # 3. no regressions
uv run python experiments/tools/validate_fixtures.py --split train --update-golden # 4. lock in
```

Before updating golden: MAPE down or held, MAE down or held, Spearman up or held, no fixture regressed beyond tolerance, change is physically grounded. **Git commit after each golden update** describing hypothesis and result.

### REFLECT

Append an entry to `AutoResearch.md` with hypothesis, change site, before/after
numbers, per-fixture regressions, physical grounding, and outcome (landed /
reverted / blocked). If MAPE dropped significantly, restart from OBSERVE — the
error landscape shifted.

## Anti-Patterns

- **Constant-tuning trap:** Adjusting efficiency floats or overhead values without understanding *why* they're wrong will overfit to the train set and won't generalize. Always ask: what physics is the model missing?
- **Overfitting to noise:** exp2 has duplicate runs for some configs. If improvement only helps one of a pair, you're fitting run variance.
- **Global constants for local problems:** If only A10G fixtures are wrong, don't change DEFAULT_COMPUTE_EFFICIENCY globally — the model is probably missing a device-specific code path.
- **Ignoring physics:** Every change must be physically grounded. No negative efficiencies or bandwidth above spec sheet.
- **Fixing symptoms not causes:** Wrong bubble_pp_delta usually stems from wrong compute timing. Fix compute first.
- **Too many knobs at once:** One conceptual change per iteration or you can't attribute results.

## Starting Hypotheses

1. **Memory-bandwidth roofline.** The compute model may lack a memory-BW ceiling. A10G (600 GB/s) and L4 (300 GB/s) are 5–11x lower than H100 (3350 GB/s). If activation-heavy stages are memory-bound on these devices, the sim under-predicts latency because it only models FLOPs. Fix: extend `compute_model.py` roofline to `max(compute_time, activation_bytes / mem_bw)`.
2. **Backward compute scaling.** A flat 2x factor may not hold across layer types (attention vs MLP vs embedding). Compare per-stage backward timing in `--verbose` against the 2x assumption. Fix: derive per-layer-type backward ratios or model recomputation overhead.
3. **GPipe fill/drain modeling.** GPipe's fill/drain bubbles amplify per-stage compute errors more than 1F1B. If GPipe fixtures are disproportionately wrong, the root cause is usually compute, not scheduling — but check whether `scheduler.py` correctly models micro-batch overlap during steady state.
4. **Cross-stage transfer contention.** With 5+ pipeline stages, multiple links may be active simultaneously and contend for shared bandwidth. If error grows with pipeline depth, the transfer model may need link-level contention (not just point-to-point latency).

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
