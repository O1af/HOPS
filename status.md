# HOPS — Status Report

**Project:** Heterogeneous Optimized Pipeline Simulator
**Team:** Jimmy Dai, Joshua Hsueh, Olaf Dsouza, Rishith Seelam

---

## What HOPS is for

A simulator that lets researchers prototype scheduling and placement algorithms for **heterogeneous** pipeline-parallel clusters, narrow the design space, and then validate the top candidates on real hardware. The bar is not bit-exact absolute prediction — it is that *relative* figures (throughput ranking across schedulers, bubble ratios, utilization, comm overhead) are trustworthy enough to guide which configurations are worth running for real.

## Where we are

HOPS is a discrete-event simulator with a solid core: seeded event engine, pipeline/hardware/latency models, pluggable scheduler registry (GPipe, 1F1B, ZeroBubble), link-contention tracking, precision scaling, ring and naive AllReduce, Chaos-Monkey failure injection, and a metrics collector that emits JSON summaries, CSV traces, and Gantt visualizations. The configuration is fully YAML-driven with device and link overrides so measured microbenchmarks can drop in without code changes.

On the instrumentation side, the `hops-tracing` branch of the Megatron-LM fork emits per-event JSONL traces (compute + P2P transfer, stage-and-microbatch attributed, CUDA-event timed) that feed directly into the HOPS importer. That means bubble ratio and per-device utilization can now be computed from real traces, not just simulator output — the calibration loop is closed end-to-end for the PP-only case.

## Validation status

One end-to-end validated scenario: **experiment 05** — 2-node H100 PP=2 on `2 x p5.4xlarge`, cross-node InfiniBand. Comparison against Megatron job 26, warmup excluded:

| Configuration | HOPS throughput | Real throughput | Error |
|---|---|---|---|
| Pre-calibration baseline | 26.74 µB/s | 12.43 µB/s | +115% |
| Link-calibrated, compute unchanged | 22.77 µB/s | 12.43 µB/s | +83% |
| Link-calibrated + compute fit (`fit1`) | 12.18 µB/s | 12.43 µB/s | **−2.0%** |

The link model is independently grounded: measured p2p (40.52 ms for 64 MB) matches HOPS (41.16 ms) within ~1.5%. The remaining error was absorbed by fitting stage compute means from Megatron logs.

Experiment 06 (PP=1 smoke) is a plumbing test only — zero bubbles, zero pipeline comm — but it confirms that the Megatron tracer, the HOPS importer, and the summary pipeline all agree on a single-GPU baseline once JIT warmup iterations are excluded.

## Progress toward the goal

- Validated one homogeneous scenario within 2% of real throughput.
- Link calibration workflow (`link_bench.py` → `overrides.links`) is general and transfers to any cluster.
- Scheduler registry means new policies drop in without touching the event engine.
- Analytical compute estimator and auto-derived activation size are in place, so new experiments no longer need a prior Megatron run to seed the simulator.
- Megatron-side telemetry is field-compatible with the HOPS importer — the same pipeline that produces sim metrics now produces ground-truth metrics.

## The next campaign

Upcoming AWS validation covers three device classes: `2 x H100`, `2 x A10G`, `2 x L4`. The plan is deliberately layered so failure modes are easy to attribute.

1. **Extend device presets.** A10G and L4 are not yet in `src/hops/presets.py`. Both need BF16 tensor-core peak, HBM/GDDR bandwidth, and memory capacity added before any run.
2. **Homogeneous baselines × 3 devices.** Clone experiment 05 for A10G and L4 at the identical model shape (hidden 3072, seq 2048, PP=2, 8 layers, bf16). This tests whether the calibration recipe itself generalizes across device types before introducing heterogeneity.
3. **Multi-shape sweep on H100.** Vary hidden size and microbatch count, refit, and check whether the per-FLOP compute constant is transferable or shape-specific. This is the single biggest open empirical question.
4. **Heterogeneous runs.** H100+A10G PP=2 and H100+L4 PP=2 — predict first, then measure. Compare both absolute error and scheduler ranking.
5. **Scheduler ablation on hetero.** 1F1B vs GPipe on each hetero pair. This is the actual research use case: does HOPS rank schedulers correctly when stage times are unequal.

## Simulator improvements alongside the campaign

- Add `tensor_numel` and `dtype_bytes` to Megatron transfer trace events so per-transfer bandwidth can be ground-truthed in-trace, without a separate `link_bench` pass.
- Strip JIT/warmup iterations in the importer (empirical threshold: discard iterations with latency ≥ 3× trailing median). Experiment 06 shows how much this matters if left in.
- Expose bubble ratio and per-device utilization deltas (sim − real) in the calibration comparison, not just throughput.
- Emit a `hetero_load_balance_ratio` (max stage compute / min stage compute) as a first-class metric — it is the quantity researchers will tune with placement algorithms.

## Checklist

### Done
- Event engine, pipeline model, hardware/link model
- YAML-driven configuration with device and link overrides
- Configurable latency distributions (normal, long-tail, heavy-tailed, Poisson)
- GPipe, 1F1B, and ZeroBubble schedulers with plugin registry
- Ring and naive AllReduce; optimizer weight-update modeling
- Metrics collector, JSON/CSV export, Gantt and 8-panel dashboards
- Link contention and peak-concurrency tracking
- P2P microbenchmark harness for link calibration
- Analytical compute-latency estimator with activation auto-derivation
- Megatron hops-tracing branch: compute + P2P transfer events, importer-compatible schema
- First real-hardware validation (exp 05, H100 PP=2) within −2%
- Link model validated within ~1.5% of measured p2p

### In progress
- A10G and L4 device presets
- Multi-shape sweep on H100 to test compute-latency transferability
- Heterogeneous validation (H100+A10G, H100+L4) on AWS
- Instrumented ground-truthing of bubble ratio and per-device utilization via imported Megatron traces

### Planned
- Transfer payload bytes in Megatron traces
- Optimizer and AllReduce events in the Megatron tracer
- Tensor-parallel and data-parallel dimensions (currently PP only)
- Interleaved 1F1B with per-virtual-chunk attribution
- Failure-injection validation: controlled slowdown/kill experiment with recovery measurement
- NUMA and intra-node bandwidth modeling for multi-GPU-per-node setups

## Metrics — met and outstanding

| Metric | Target | Status |
|---|---|---|
| Absolute throughput accuracy (homogeneous) | within 10% | **Met** (−2.0%) |
| Link-transfer prediction | within 5% of measured | **Met** (~1.5%) |
| Relative ranking of configs (bubble, util, comm) | directionally correct | **Met** qualitatively; not yet ground-truthed |
| Absolute bubble ratio accuracy | within 15% | Pending imported-trace comparison |
| Absolute device utilization accuracy | within 10% | Pending imported-trace comparison |
| Multi-shape transferability of compute latency | stable ±20% across model sizes | **Open** — next H100 sweep |
| Heterogeneous cross-device accuracy | within 15% | **Open** — blocked on A10G/L4 presets + runs |
| Scheduler ranking on hetero | top-k stable | **Open** — validation campaign deliverable |
| Failure-injection accuracy | recovery within 20% | Not yet attempted |
| TP/DP support | functional | Not yet implemented |

## Honest assessment

The simulator is credible for the one homogeneous scenario we have validated, and the architecture is right for the job. The two open questions are empirical, not architectural: does the compute-latency abstraction transfer across model shapes on the same device, and does it transfer across device types. The first we answer with an H100 sweep; the second is the point of the A10G and L4 runs. The upcoming campaign is designed so that if either assumption fails, we see it at the layer where it fails — and if both hold, HOPS becomes genuinely useful for scheduling research on real heterogeneous clusters.
