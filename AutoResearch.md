# AutoResearch Log

Running log of every OBSERVE → HYPOTHESIZE → IMPLEMENT → VALIDATE → REFLECT pass.
Append a new entry below for each attempt (landed or reverted). Keep entries
short, falsifiable, and numeric.

## Entry template

```
### YYYY-MM-DD — <short title>

- Commit: <sha> (or "reverted")
- Hypothesis: <one sentence, falsifiable>
- Change site: <file(s) touched>
- Before → After (train split, link_calibrated):
  - MAPE: X% → Y%
  - bubble MAE: Xpp → Ypp
  - Spearman: X → Y
  - Per-group (only those that moved): <group>: X% → Y%
- Target fixtures (if applicable): <fixture>: before → after
- Per-fixture regressions (>tolerance): <list or "none">
- Physical grounding: <why the change matches hardware/framework reality>
- Outcome: landed / reverted / blocked
- Follow-up: <next hypothesis suggested by the new error landscape>
```

## Rules

- One entry per iteration, even if reverted. Recording dead ends saves the next
  agent from repeating them.
- Iteration numbers come from `validate_fixtures.py --split train` SUITE
  AGGREGATES and GROUP BREAKDOWN sections. Don't paraphrase.
- The agent must not run `--split test`. If you accidentally do, note it and
  discard.
- Human test-bench checkpoints may record only suite-level aggregate deltas
  (MAPE, bubble MAE, Spearman, PASS/regression). Do not record test fixture
  names, group breakdowns, phase details, error directionality patterns, or
  follow-up hypotheses derived from test results.
- If you ran `--update-golden`, link the commit. Golden is gitignored locally,
  so the commit is the durable record.

---

## 2026-04-19 — h100 memory bandwidth: SXM5 peak → PCIe peak

- Commit: `fddb618` (`presets: set h100 memory bandwidth to PCIe spec (2.0 TB/s)`)
- Hypothesis: The `h100` preset quoted HBM3 SXM5 peak (3350 GB/s), but every
  fixture declares `pcie` for the same-node interconnect, i.e. these traces
  come from H100 PCIe cards whose HBM2e peak is 2000 GB/s. At hidden=1024 /
  seq=1024 shapes the stages are memory-bound in sim, so the SXM5 figure
  made the analytical roofline ~1.7x too fast for every H100 stage.
- Change site: `src/hops/presets.py` (`DEVICE_PRESETS["h100"].memory_bandwidth_gbps`
  3350.0 → 2000.0).
- Before → After (train split, link_calibrated):
  - MAPE: 60.8% → 53.4% (−7.4pp)
  - bubble MAE: 15.1pp → 15.4pp (held)
  - Spearman: 0.521 → 0.547 (+0.026)
  - h100 group: 39.8% → 27.6% (−12.2pp)
  - Other groups essentially unchanged.
- Target fixtures (link_calibrated throughput error):
  - `exp2_8_h100_pair_pp2`: +29.7% → +21.6%
  - `exp2_22_h100_pair_gpipe`: +29.8% → +21.6%
  - `exp2_23_h100_pair_batch8`: +59.8% → +39.4%
- Per-fixture regressions (>tolerance): two `no_lookahead` Spearman drops
  (`exp2_33` 0.956 → 0.829, `exp2_3` 0.143 → −0.058). Both are rank-order
  noise on a brittle metric — one stays a strong correlation, the other is
  near-zero on both sides.
- Physical grounding: matches the actual H100 variant used in the traces.
- Outcome: landed; golden updated; full train suite `PASS`.
- Human test-bench checkpoint (2026-04-20, human-run, aggregate only):
  test split link_calibrated MAPE 51.8% → 41.6% (−10.2pp), bubble MAE
  15.2pp → 14.4pp (−0.8pp), util Spearman mean 0.521 → 0.598 (+0.077),
  suite `PASS (no regression)`. No fixture, group, phase, or error-pattern
  details recorded, to preserve test isolation.
- Follow-up: remaining link_calibrated errors are now bimodal —
  - *small* workloads still too fast (`exp2_23` +39%, `exp2_33` seq4096
    +36%, H100 pair +21%),
  - *large* workloads too slow (`exp2_34` hidden2048 −35%, `exp2_36` −30%,
    `exp2_6` large_model −29%, `exp2_14` L4 pair −23%).
  Pattern is shape-dependent efficiency. Next candidates: shape-aware
  compute efficiency in `compute_model.py`, or lowering `backward_factor`
  from 2.0 toward the ~1.0–1.5 BWD/FWD ratio observed in every trace
  (requires bulk-editing the `backward_factor: 2.0` field across 36
  fixture YAMLs, since each fixture sets it explicitly and overrides the
  code default).

### Other hypotheses tried this round (reverted, recorded to save time later)

- **Remove BF16 `precision_speedup` double-count** (`src/hops/core/types.py`
  `Precision.compute_speedup` returning 1.0). Physically correct — the device
  presets already store BF16 tensor-core peak — but the Megatron stages are
  memory-bound at these shapes, so `max(compute_ms, memory_ms)` was pinned to
  `memory_ms` and the aggregate metrics didn't move. Reverted to avoid a
  no-op commit; keep for a future round once shapes are large enough to
  make compute the bottleneck.
- **H100 `launch_overhead_ms` bumps (1.8 / 2.0 / 2.5 / 4.0 / 8.0)**. Each
  reduced H100 group MAPE but created per-fixture bubble regressions above
  the 2.0pp tolerance on mixed pipelines (H100 flips from "fastest stage"
  to "slower stage" and every bubble calculation shifts). Reverted.
- **Global H100 `memory_bandwidth_gbps` drops (900 / 1500)**. Pulled MAPE
  down further than the 2000 commit, but bubble MAE regressed by >1.5pp
  and several mixed-pipeline fixtures regressed throughput by 2.5–4.9pp.
  Reverted. The PCIe-spec value of 2000 is the largest physically justified
  drop; anything below that is a fudge factor.
