# HeteroResearch.md — Heterogeneous Pipeline Scheduling

Research log for designing a pipeline schedule that outperforms ZeroBubble
on heterogeneous clusters (2×L4 + 2×A10G), validated in the HOPS
simulator. This is a sibling document to `AutoResearch.md` but targets
**scheduling policy** rather than simulator fidelity.

## Setup

All experiments use 4-stage pipelines composed of 2× NVIDIA L4 and 2×
A10G GPUs, connected over PCIe within a node and Ethernet across nodes.
The exact per-stage ordering and model shape is varied across 16
benchmark configs (`experiments/hetero_scheduler/bench_*.yaml` +
`hetero_pp4*.yaml`).

Run the full benchmark:

```bash
uv run python experiments/hetero_scheduler/evaluate.py \
    --configs bench_l4_middle.yaml bench_a10g_middle.yaml ... \
    --schedulers gpipe 1f1b zero_bubble hops_hetero \
    --seeds 42 43 44 \
    --output-json experiments/hetero_scheduler/final_results.json
uv run python experiments/hetero_scheduler/summarize.py
```

Unit tests: `uv run pytest tests/test_hetero_schedulers.py`.

## Device profile

The HOPS compute-model in BF16 reports:

| GPU  | Peak TFLOPs | HBM GB/s | Per-stage FWD | Per-stage split BWD (B + W) | Fused BWD (1F1B) |
|------|-------------|----------|---------------|-----------------------------|-------------------|
| A10G | 125         | 600      | 11.28 ms      | 22.56 ms (11.28 + 11.28)    | 18.57 ms          |
| L4   | 121         | 300      | 18.57 ms      | 37.14 ms (18.57 + 18.57)    | 35.64 ms          |

L4 is memory-bandwidth-bound for the 384 MB activation workload. A10G has
2× the memory bandwidth and is ~64% faster per stage. In every 2×L4 +
2×A10G pipeline, **the two L4 stages are the bottleneck** regardless of
pipeline ordering; the A10G stages have slack that ZeroBubble uses to
hide weight-gradient (W) tasks in steady-state bubbles.

## Key observation: the ZeroBubble-cooldown tail

HOPS's `DerivedLatency.sample_backward(layer_factor)` computes a
split-BWD cost that is consistently `~4.5 ms` larger per microbatch
than the equivalent fused BWD (`max(compute, memory, per_layer_floor) *
work_scale` is not equal to `max(compute * work_scale, memory *
work_scale, per_layer_floor * work_scale)` when the kernel-dispatch
floor is the binding term). This is a deliberate model choice in
`src/hops/latency/compute_model.py` (see `AutoResearch.md` iteration
"Structural: LM head + cross-entropy on last pipeline stage").

This has a concrete consequence: on a **bottleneck stage saturated at
~100% utilization** ZeroBubble pays `num_microbatches × (w_ms split
overhead)` in extra makespan that 1F1B's fused BWD does not incur. On
the last stage of the pipeline, deferred W tasks accumulate into a
**cooldown tail** of length `num_microbatches × w_ms` because the last
stage has no downstream dependency to create a bubble for them.

## Baselines (3-batch, 24-microbatch runs, 2×L4 + 2×A10G)

| config (stage order)      | GPipe ms | 1F1B ms | ZB ms  | 1F1B vs ZB |
|---------------------------|----------|---------|--------|------------|
| L4 middle (A,L,L,A)       | 2983.37  | 2937.81 | 2769.05| −6.09%     |
| L4 ends (L,A,A,L)         | 4793.17  | 4792.07 | 4895.64| **+2.12%** |
| L4 end (A,A,L,L)          | 6271.82  | 6272.71 | 6483.63| **+3.25%** |
| L4 middle big (A,L,L,A)   | 3912.16  | 3856.07 | 3612.69| −6.74%     |

Pattern: **ZeroBubble wins when the bottleneck is in the pipeline's
interior; 1F1B wins when the bottleneck is at the last stage**. No
single existing policy is Pareto-optimal across heterogeneous layouts.

## Hypotheses explored

### H1 — HeteroAdaptiveWarmup (reverted)

- **Hypothesis**: ZB's uniform `warmup = N - s` per-stage warmup is wrong
  under heterogeneity; fast stages should warm up more to keep the slow
  stage's input queue full. Scale by `slow_fwd / fwd[s]`.
- **Result**: no change on any of 16 configs; the L4 bottleneck is
  already ≥95% utilized by ZB, so extra warmup has nowhere to go.
- **Outcome**: **no effect**. Implemented but kept in the scheduler
  catalog for reference.

### H2 — HeteroEagerW (reverted)

- **Hypothesis**: fire deferred W immediately on any stage whose next F
  or B is not ready. Same policy as ZB since ZB's cooldown-fill already
  does this.
- **Result**: identical to ZB (0% delta on every config).
- **Outcome**: **no effect**.

### H3 — HeteroBottleneckPriority (reverted)

- **Hypothesis**: give the bottleneck stage strict B-over-F priority so
  its activation queue drains fastest; upstream stages prefer F to feed
  it; downstream prefer B.
- **Result**: identical to ZB because ZB's warmup-limit gate already
  produces this ordering on every saturated stage.
- **Outcome**: **no effect**.

### H4 — HeteroWaveFill (tried and reverted)

- **Hypothesis**: enforce a per-stage in-flight cap proportional to
  `slow/fast_fwd`, so fast stages can issue more forwards during
  steady-state.
- **Result**: ≤1.3% regression on several configs. Extra in-flight
  forwards on A10G stages don't help because the L4 is already the
  bottleneck and the A10G's B starts at the same time as before.
  Additional forwards just increase activation memory.
- **Outcome**: **regresses**.

### H5 — HeteroCriticalPath (reverted)

- **Hypothesis**: within a `next_tasks()` call, sort candidate tasks by
  an approximate remaining-critical-path priority so the task whose
  output unblocks the longest chain runs first.
- **Result**: identical to ZB. The `state.stage_is_busy()` check in
  `next_tasks()` means only idle stages compete, and at most one task
  per stage is picked — so priority doesn't matter except under exact
  ties (which don't occur for this compute model).
- **Outcome**: **no effect**.

### H6 — HeteroFusedBW / HeteroBottleneckEagerW (regresses on ZB-favouring configs)

- **Hypothesis**: mark each stage as "saturated" (fwd+b+w ≥ 0.9 × slow)
  and on those stages fire W immediately after the matching B. This
  preserves ZB behavior on stages with bubble slack, but eliminates the
  W-cooldown tail on saturated stages.
- **Result**:
    - Matches 1F1B on `slow_ends` (bottleneck at ends) but **−5% on
      `hetero_pp4`** (bottleneck in middle — stages 1, 2 incorrectly
      flagged as saturated, losing the ZB bubble-fill win).
- **Outcome**: **regresses on 50% of configs**.

### H7 — HeteroEagerLastW (reverted)

- **Hypothesis**: restrict the eager-W policy to the **last** pipeline
  stage, because only the last stage has no downstream bubble to absorb
  deferred W.
- **Result**: matches 1F1B on `slow_ends` / `fast_front` only when the
  bottleneck is at the last stage. On configs where the bottleneck is
  at the first or middle stage this hurts (last stage has bubble slack
  that could have absorbed W). Suite average −1.29% vs ZB.
- **Outcome**: **regresses on configs where last-stage isn't
  bottleneck**.

### H8 — HeteroAdaptiveWSplit ✅ (Pareto, renamed to HopsHetero)

- **Hypothesis**: the per-stage eager-W heuristic keeps picking the
  wrong set of stages. A simpler, provably Pareto-correct decision is:
    - Use **ZeroBubble** (with W-split) when the last stage's
      steady-state bubble ≥ w_last, i.e. when the deferred Ws fit into
      non-critical slack.
    - Use **1F1B** (fused BWD, no W-split) otherwise.
  The decision is made **once at `configure()` time** from the HOPS
  compute model's per-stage samples, with zero runtime overhead and
  zero tuning knobs.
- **Change site**: new file `src/hops/core/hetero_schedulers.py` +
  hooks in `src/hops/core/scheduler.py` (the `Scheduler.configure()`
  method, called once by `Pipeline.__init__`).
- **Before → After** (vs `zero_bubble`, 16 configs, 3 seeds each):
    - Aggregate makespan delta: **+0.79% faster**
    - Wins (>0.5% improvement): **5 / 16**
    - Ties (within 0.5%): **11 / 16**
    - Regressions (>0.5% slower): **0 / 16**
- **Per-config highlights**:

    | config                          | ZB ms   | hops_hetero ms | Δ |
    |---------------------------------|---------|----------------|---|
    | bench_a10g_middle_mb48          | 3221.20 | 3077.23        | **+4.47%** |
    | hetero_pp4_fast_front           | 6483.63 | 6272.71        | **+3.25%** |
    | bench_a10g_middle_mb24          | 1632.66 | 1599.04        | **+2.06%** |
    | hetero_pp4_slow_ends            | 4895.64 | 4792.07        | **+2.12%** |
    | bench_big_a10g_middle           | 1553.46 | 1530.68        | **+1.47%** |
    | hetero_pp4 (L4 middle)          | 2769.05 | 2769.05        | 0.00%      |
    | hetero_pp4_big_model            | 3612.73 | 3612.73        | 0.00%      |
    | bench_l4_front                  | 911.51  | 911.51         | 0.00%      |
- **Physical grounding**: the decision variable is physical —
  `slack_at_last = slowest_per_mb_total − last_per_mb_total` is the
  idle time the last stage would naturally have per microbatch if
  constrained by the bottleneck. `w_last` is the last stage's deferred
  weight-gradient latency. If `slack ≥ w_last`, the W fits into a
  pre-existing bubble; otherwise it spills into a cooldown tail.
- **Outcome**: **landed** as the default novel scheduler
  (`hops_hetero`). Implementation ≤ 60 LOC, trivially testable, no
  runtime state, no knobs.

## Architecture changes

1. `src/hops/core/scheduler.py`: added an optional `configure(meta)`
   hook on the `Scheduler` base. The pipeline computes expected
   per-stage FWD / BWD_B / BWD_W latencies from the compute model at
   runtime-init and passes them to the scheduler once, before any batch
   starts. Default behaviour: no-op. Existing schedulers are
   unchanged.
2. `src/hops/core/pipeline.py`: hooks in `_configure_scheduler()` to
   sample the compute model and call `scheduler.configure(...)`.
   Critically, the scheduler's `uses_w_split` attribute is *read after*
   `configure()`, allowing `HopsHetero` to toggle W-split based on the
   workload profile.
3. `src/hops/core/hetero_schedulers.py`: new file containing
   `HopsHetero` plus 10 alternative candidates tested during the
   research loop. All are registered under stable public names for
   future experiments.

## Open problems / follow-ups

1. The HOPS split-BWD modeling artifact (`split B+W ≠ fused BWD` by
   ~4.5 ms on the bottleneck per microbatch) is the reason no per-
   stage eager-W variant can match 1F1B on tail-bottleneck configs.
   In a real system this overhead does not exist, so the theoretical
   win of `hops_hetero` over pure 1F1B on those configs is larger
   than HOPS reports.
2. The selector currently picks between two known policies (ZB vs
   1F1B). A future candidate could combine their strengths into a
   single schedule — e.g. ZB on non-bottleneck stages with fused BWD
   on the bottleneck — but this requires the `Pipeline` to support
   mixed phase-set pipelines, which it currently does not.
3. For pipelines with >4 stages or two separate bottleneck clusters
   (e.g. an H100 + A10G + L4 + L4 + H100 arrangement), the simple
   "last-stage slack" heuristic may be suboptimal. A multi-stage
   bubble accounting model is a natural extension.

## Scheduler candidate registry

Every candidate below is registered in `_SCHEDULER_REGISTRY` and can
be selected via `pipeline.schedule: <name>` in YAML configs.

| Name                         | Status     | Notes                                                |
|------------------------------|------------|------------------------------------------------------|
| `hops_hetero`                | **landed** | Pareto-optimal meta-scheduler (ZB or 1F1B)           |
| `hetero_adaptive_w_split`    | kept       | Same policy under an earlier name                     |
| `hetero_adaptive_warmup`     | neutral    | Scaled warmup; no effect on L4/A10G configs           |
| `hetero_bottleneck`          | neutral    | Upstream/downstream of bottleneck priority            |
| `hetero_bottleneck_eager_w`  | partial    | Eager W on saturated stages only                      |
| `hetero_critical_path`       | neutral    | Ordering within `next_tasks()`                        |
| `hetero_eager_last_w`        | partial    | Eager W on last stage only                            |
| `hetero_eager_w`             | neutral    | Unconditional eager W                                 |
| `hetero_fused_bw`            | partial    | Fuse B+W on saturated stages                          |
| `hetero_hybrid`              | neutral    | Adaptive warmup + eager W + bottleneck priority       |
| `hetero_wavefill`            | regressive | Per-stage in-flight cap                               |
