# Heterogeneous pipeline-parallel scheduling (L4 / A10G)

Research log for scheduling on a budget of up to 2× L4 + 2× A10G, same-node `pcie`, cross-node `ethernet`, using HOPS only. **No** reads or runs against `fixtures/test/`.

**Methodology:** OBSERVE → HYPOTHESIZE → IMPLEMENT → VALIDATE → REFLECT (aligned with `AGENTS.md`). Prior fixture-calibration work is in `AutoResearch.md` (not repeated here).

**Tooling:** Synthetic sweeps via `uv run python experiments/tools/bench_hetero_sched.py` (builds in-memory YAML, no golden files). Full sim: `uv run python main.py --config <path>.yaml --no-viz --summary-json /tmp/result.json`.

**Implemented policy:** `pipeline.schedule: heterogeneous_hops` — registered in `src/hops/core/scheduler.py`, constructed in `build_runtime()` with full `AppConfig` + `Topology` + `PresetRegistry` passed to `make_scheduler()` so the policy can use per-stage analytical work and device presets **without** changing `TimingModel` or `ComputeModel`.

---

## OBSERVE — Phase 1 baselines (synthetic)

**Setup:** BF16, `activation_mb: 40`, `backward_factor: 2.0`, `backward_split.enabled: true`, `activation_grad_fraction: 0.5`, `optimizer.enabled: false`, interconnect `pcie` / `ethernet`, constant zero jitter. Stages use analytical `tflop` + `memory_mb` + fixed efficiencies.

**Metrics:** `throughput` = microbatches/s (`summary.throughput.per_s`), `bubble%` = `100 * bubble_ratio`, `latency_ms` = mean e2e per microbatch, `peak_mem_mb` = max over devices, `util` = per-stage utilization string from summary.

### PP=2, uniform FLOP (4 TF / stage), cross-node pair

| scenario | mb | scheduler | thr/s | bubble% | latency_ms | peak_mem_mb |
|----------|---:|-----------|------:|--------:|-----------:|------------:|
| L4→A10G | 8 | gpipe | 11.74 | 12.93 | 417.76 | 6160 |
| L4→A10G | 8 | 1f1b | 12.05 | 10.66 | 277.93 | 6100 |
| L4→A10G | 8 | zero_bubble | 17.36 | 3.01 | 163.23 | 6080 |
| A10G→L4 | 8 | zero_bubble | 16.96 | 5.69 | 180.04 | 6080 |
| L4→A10G | 12 | zero_bubble | 18.03 | 1.54 | 160.03 | 6080 |
| L4→A10G | 16 | zero_bubble | 18.38 | 1.51 | 158.43 | 6080 |
| L4→A10G | 24 | zero_bubble | 18.75 | 1.51 | 154.70 | 6080 |

(Additional rows for mb 12/16/24 and `gpipe`/`1f1b` mirror the bench script output; `heterogeneous_hops` matches `zero_bubble` on these rows — see Phase 3.)

### PP=4, four GPUs same node, uniform 2.5 TF / stage

Representative sample (full grid in script output):

| ordering | mb | zero_bubble thr/s | bubble% |
|----------|---:|------------------:|--------:|
| L4,L4,A10G,A10G | 8 | 20.04 | 10.41 |
| L4,L4,A10G,A10G | 24 | 25.87 | 2.87 |
| A10G,L4,A10G,L4 | 8 | 20.45 | 12.63 |
| A10G,L4,A10G,L4 | 24 | 26.18 | 5.48 |

**Observation:** On these analytical configs, `zero_bubble` dominates `gpipe` and `1f1b` on throughput and bubble. Ordering (slow-first vs interleaved) shifts absolute throughput but ZeroBubble remains the strong baseline.

---

## HYPOTHESIZE — Phase 2 (falsifiable directions)

1. **Latency-aware warmup depth:** If warmup forwards at stage `i` scale with downstream/upstream compute imbalance, a slow stage feeding a fast stage should issue **more** warmup F before B than the homogeneous `num_stages - i` rule. *Falsified on current sim for PP2/PP4 sweeps:* extra warmup capped at `num_microbatches` did not change the event order vs ZeroBubble when downstream was faster (identical metrics).

2. **Latency-aware W deferral:** Defer `BACKWARD_W` on stages with below-median estimated compute-per-microbatch; run W eagerly on the critical stage. *Not differentiated in HOPS for tested configs:* deferred W still issues in the same global order as ZeroBubble (same throughput/bubble/latency).

3. **Critical-path stage priority:** Reorder ready-task arbitration so the slowest stage always wins. *Partially redundant here:* pipeline already serializes per stage; cross-stage priority is dominated by existing F/B readiness and transfer timing.

4. **Topology-aware F vs B tie-break:** Prefer forwards on a “feeder” when it is faster than its downstream neighbor. *Tested and reverted:* swapping F before B for fast→slow feeders **reduced** throughput (~0.6–2.0%) on `pp4_a10gl4a10gl4_uniform` at mb 8/12 because it delayed backward pressure and lengthened the critical path in this discrete simulator.

5. **Variable-granularity W packing:** Would require scheduler support for partial-W tasks or finer task keys; **not implemented** (would extend `Phase` / pipeline beyond “scheduling policy only” as stated).

6. **Backward-first on fast stages to save memory:** Lower warmup on fast stages could reduce peak activations. **Not implemented** — risks violating the same in-flight bounds as 1F1B/ZeroBubble and needs memory-model coupling beyond schedule tweaks.

7. **Dynamic steady-state from estimated queue depth:** Would need simulator-exposed queue depth or predicted transfer backlog. **Not implemented** — not available on `PipelineState` without engine changes.

---

## IMPLEMENT — Phase 3

### Code

- `HeterogeneousHopsScheduler` in `src/hops/core/scheduler.py`:
  - `build_stage_compute_weights()` — deterministic forward-ms estimate per stage (mirrors analytical roofline: TFLOP, memory MB, device FLOPS and memory BW from presets + overrides, NUMA penalty scales, optional `pipeline.model` layer-count floor).
  - Warmup: `base = num_stages - stage` plus `round(downstream_weight / upstream_weight - 1)` when downstream is heavier, capped by `num_microbatches`.
  - W rule: stages slower than the median weight run W in the ZeroBubble slot; faster stages defer W to a second pass when the engine would otherwise stall.
- `make_scheduler(..., app=, topology=, registry=)` — `heterogeneous_hops` requires these kwargs.
- `src/hops/runtime.py` — passes `config`, `topology`, `registry` into `make_scheduler`.
- `max_in_flight_count()` — treats `heterogeneous_hops` like 1F1B/ZeroBubble for memory validation.
- `experiments/tools/bench_hetero_sched.py` — reproducible sweeps.
- `tests/test_scheduler.py` — asserts missing context raises for `heterogeneous_hops`.

### Config

Use:

```yaml
pipeline:
  schedule: heterogeneous_hops
```

---

## VALIDATE — Phase 3 & 4

**Unit tests:** `uv run pytest tests/ --ignore=tests/test_validation_regression.py` → **166 passed**.

**Note:** `tests/test_validation_regression.py` fails on this workspace **without** these scheduler changes (golden tolerance on `exp3_01` bubble); excluded for this iteration’s green run.

**Stress:** `bench_hetero_sched.py` over all built-in scenarios and `microbatches ∈ {8,12,16,24,32}`: **max throughput delta vs `zero_bubble` = 0%** on every cell (heterogeneous_hops matches ZeroBubble metrics exactly after reverting the harmful F-first tie-break).

---

## REFLECT — Iterations

### Iteration 1 — heterogeneous_hops (warmup + W deferral by stage weight)

| Hypothesis | If true, `heterogeneous_hops` would beat `zero_bubble` on throughput for skewed L4/A10G configs at multiple microbatch counts. |
| Results (sample) | PP2 L4→A10G uniform mb=8: ZB thr=17.36/s, het=17.36/s, Δ=0%. PP4 A10G/L4 interleaved mb=12: ZB=23.32/s, het=23.32/s, Δ=0%. |
| Delta vs ZeroBubble | 0% throughput, 0pp bubble, same latency and peak memory on full bench grid. |
| Analysis | Readiness rules in HOPS are strict; for every tested state, the new policy selects the **same** next task set as `ZeroBubbleScheduler`. Warmup extensions never change the feasible schedule because microbatch count and transfer ordering already constrain the critical path identically. |
| Outcome | **Landed** as a registered policy and wiring for future experiments, but **did not** meet the >5% throughput success criterion. |

### Iteration 2 — feeder forward first (reverted in code)

| Hypothesis | On fast→slow edges, issuing an extra forward before backward B reduces downstream starvation vs index-based ZeroBubble. |
| Results | `pp4_a10gl4a10gl4_uniform` mb=8: throughput **−0.60%** vs ZB; mb=12: **−2.0%**; bubble unchanged. |
| Outcome | **Reverted** (not shipped). |

---

## Phase 5 (optional) — real traces

Not executed here (train fixture inspection allowed; user may run `uv run python experiments/tools/inspect_fixture.py --fixture exp2_17_a10g_l4_pair_pp2_run147`). Scheduling-only changes do not alter trace replay calibration paths.

---

## Success criteria check

| Criterion | Result |
|-----------|--------|
| Primary: >5% throughput vs ZeroBubble on ≥3 microbatch counts | **Not met** on synthetic L4/A10G suite |
| Secondary: bubble / util / memory | No improvement vs ZeroBubble (identical numbers) |
| Stretch: generalize across PP depths | Policy is safe (matches ZB) but **no gain** |

---

## Follow-up (next agent)

1. **Expose richer state to schedulers** (engine change, not done here): e.g. pending transfer queue depth per link, or “downstream stage idle” hints — without that, many heterogeneous heuristics collapse to ZeroBubble’s decisions under HOPS readiness rules.

2. **Topology-aware tie-breaking** needs a metric that differs from per-stage compute weights alone — e.g. precomputed per-edge transfer time included in a scheduler-only scoring model (still scheduling-only if computed from config + presets in the scheduler factory).

3. **Calibrated heterogeneity:** sweep `memory_mb` / `tflop` so stages are not near-identical roofline times; the current uniform configs make ZB nearly optimal.
