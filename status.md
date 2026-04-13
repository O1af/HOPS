# HOPS — Status Report

**Project:** Heterogeneous Optimized Pipeline Simulator
**Team:** Jimmy Dai, Joshua Hsueh, Olaf Dsouza, Rishith Seelam

---

## Where we are

HOPS is a discrete-event simulator for pipeline-parallel training. The core is in place: event engine, pipeline/hardware/latency models, a pluggable scheduler registry, link-contention tracking, seeded RNG for determinism, and a metrics collector that emits JSON summaries, CSV traces, and Gantt-style visualizations. Configuration is fully YAML-driven, and device/link overrides allow measured microbenchmark data to flow directly into the simulator without code changes.

The architecture is solid enough that extending it — adding a new schedule, a new link model, a new failure mode — is a config or single-module change rather than a core rewrite.

## Validation status

We have one end-to-end validated scenario: **experiment 05**, a 2-node H100 PP=2 baseline on p5.4xlarge (single H100 per node, cross-node InfiniBand). The comparison against a real Megatron-LM run (slurm job 26) gives the following picture after excluding JIT/warmup iterations:

| Configuration | HOPS throughput | Real throughput | Error |
|---|---|---|---|
| Pre-calibration baseline | 26.74 µB/s | 12.43 µB/s | +115% |
| Link-calibrated, compute unchanged | 22.77 µB/s | 12.43 µB/s | +83% |
| Link-calibrated + compute fit (`fit1`) | 12.18 µB/s | 12.43 µB/s | **−2.0%** |

The link model is well-grounded: measured p2p bandwidth (40.52 ms for 64 MB) matches the HOPS prediction (41.16 ms) to within measurement noise. The remaining calibration work is the compute-latency estimator.

**Current constraint:** we are compute-bound on the validation side. We only have H100 access right now, which limits us to homogeneous validation. An AWS quota request for A100 instances is pending; once it clears, we will run the first genuinely heterogeneous cross-device validation (H100 + A100 PP=2).

## Metrics HOPS reports

- Throughput (micro-batches/s, /ms)
- End-to-end latency (mean, p50, p99)
- Pipeline bubble ratio
- Per-stage and per-device time utilization
- Communication overhead (transfer_ms / compute_ms)
- Per-link transfer utilization and peak concurrency
- Contended transfer fraction
- Peak memory per device
- Failure impact (count, downtime, lost work) — wired but not yet validated

Throughput is validated against reality. The breakdown metrics (bubble, utilization, comm overhead) are internally consistent but not yet ground-truthed against instrumented runs.

## End goal

A simulator that researchers use to prototype scheduling and placement algorithms for **heterogeneous** pipeline clusters without needing to run Megatron every time — narrowing a large design space down to a handful of candidates that are actually worth running on real hardware. The bar is not bit-exact absolute prediction; it is that *relative* figures (bubble %, utilization, comm overhead, throughput ranking across schedulers) are trustworthy enough to guide decisions.

## Progress toward that goal

- Validated single homogeneous config within 2% of real throughput.
- Link characterization methodology (`link_bench.py` → YAML overrides) is general and transfers to any cluster.
- Modular scheduler registry means a new 1F1B variant can be dropped in without touching the event engine.
- Calibration workflow (p2p benchmark → link overrides → compute refit) is documented in experiment 05 and reproducible.

## Checklist

### Done
- [x] Event engine, pipeline model, hardware/link model
- [x] YAML-driven configuration with device/link overrides
- [x] Configurable latency distributions (normal, long-tail, heavy-tailed, Poisson)
- [x] 1F1B scheduling policy
- [x] Metrics collector + JSON/CSV export
- [x] Gantt and 8-panel summary visualizations
- [x] Link contention and peak-concurrency tracking
- [x] P2P microbenchmark harness for link calibration
- [x] First end-to-end real-hardware validation (exp 05, H100 PP=2)
- [x] Link model validated to within <2% of measured p2p

### In progress
- [ ] AWS quota approval for A100 instances (blocking hetero validation)
- [ ] Multi-config validation sweep on H100 (varying hidden size, µB count, PP depth) to test whether compute latency is a transferable per-FLOP quantity or an overfit point estimate
- [ ] Instrumented Megatron baseline (Nsight / built-in timers) to ground-truth bubble %, device utilization, and communication-overhead breakdowns

### Planned
- [ ] Analytical compute-latency estimator (FLOPs × device peak × efficiency) so users do not need a prior Megatron run to seed the simulator
- [ ] Auto-derived activation size from pipeline config (removes a manual calibration step)
- [ ] Heterogeneous validation: H100 + A100 PP=2 once quota lands
- [ ] Additional scheduling policies: GPipe, interleaved 1F1B, Zero Bubble
- [ ] Optimizer step / AllReduce modeling (currently `optimizer.enabled: false`)
- [ ] Tensor-parallel and data-parallel dimensions (at present PP only)
- [ ] Failure-injection validation: controlled slowdown/kill experiment with recovery measurement
- [ ] NUMA and intra-node bandwidth modeling for multi-GPU-per-node setups

## Metrics — met and outstanding

| Metric | Target | Status |
|---|---|---|
| Absolute throughput accuracy (homogeneous) | within 10% | **Met** (−2.0%) |
| Link-transfer prediction | within 5% of measured | **Met** (~1.5% error on 64 MB) |
| Relative ranking of configs (bubble, util, comm) | directionally correct | **Met** qualitatively; not ground-truthed |
| Absolute bubble ratio accuracy | within 15% | Pending instrumented run |
| Absolute device utilization accuracy | within 10% | Pending instrumented run |
| Heterogeneous cross-device accuracy | within 15% | Blocked on A100 quota |
| Multi-config transferability of compute latency | stable ±20% across model sizes | Blocked on validation sweep |
| Failure-injection accuracy | recovery-time within 20% | Not yet attempted |
| TP/DP support | functional | Not yet implemented |

## Honest assessment

The simulator is credible for the single homogeneous scenario we have validated, and the architecture is right. The two biggest open questions are both empirical rather than architectural: does the compute-latency abstraction transfer across model configurations on the same hardware, and does it transfer across heterogeneous device types. The first we can answer on H100 alone in the next validation sweep; the second is gated on A100 availability. Everything else on the checklist is scoped work, not a research risk.
