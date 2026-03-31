# Heterogeneous Cluster Validation Experiments

These assets are designed for a small ParallelCluster validation campaign where:

- one compute node is a `p5.48xlarge` (H100)
- one compute node is a `p4d.24xlarge` (A100)
- you want the real Megatron execution topology to match the HOPS topology as closely as possible

## Scenarios

1. `01_h100_baseline_pp2`
   Single-node H100 baseline using 2 GPUs and `PP=2`.
   Purpose: calibrate per-stage H100 timing without cross-node effects.

2. `02_a100_baseline_pp2`
   Single-node A100 baseline using 2 GPUs and `PP=2`.
   Purpose: calibrate per-stage A100 timing without cross-node effects.

3. `03_hetero_even_pp4`
   Mixed H100+A100 run using 2 GPUs from each node and `PP=4`.
   Purpose: measure the naive heterogeneous pipeline with an even layer split.

4. `04_hetero_weighted_pp4`
   Mixed H100+A100 run using 2 GPUs from each node and `PP=4`.
   Purpose: measure whether giving more layers to the H100 stages improves throughput.

## Files

- `link_bench.py`
  NCCL-based point-to-point and all-reduce microbenchmark. Use this to fit HOPS `overrides.links`.
- `*/hops.yaml`
  Initial HOPS template for the scenario. The explicit stage means are placeholders that should be updated after the first calibration run.
- `*/run.slurm`
  Slurm batch script intended for submission from the ParallelCluster head node.

## Prerequisites

- This repo is visible on the compute nodes through shared storage.
- `MEGATRON_DIR` points to a shared Megatron-LM checkout.
- The nodes have a working CUDA/NCCL stack.
- The head node can submit Slurm jobs.
- For the heterogeneous scripts, the allocation contains exactly one H100 node and one A100 node.

## Submission Examples

Single-node H100 baseline:

```bash
sbatch -p p5 experiments/01_h100_baseline_pp2/run.slurm
```

Single-node A100 baseline:

```bash
sbatch -p p4d experiments/02_a100_baseline_pp2/run.slurm
```

Heterogeneous scenarios:

```bash
sbatch experiments/03_hetero_even_pp4/run.slurm
sbatch experiments/04_hetero_weighted_pp4/run.slurm
```

If your cluster does not expose both node types from one partition, keep the inner `torchrun` logic from the heterogeneous scripts but translate the allocation to your site-specific Slurm constraints.

## Outputs

Each `run.slurm` creates:

- `output/<job_id>/hops_summary.json`
- `output/<job_id>/hops_trace.csv`
- `output/<job_id>/hops_stdout.txt`
- `output/<job_id>/node_inventory.txt`
- one or more Megatron logs

## Plan

Proceed in two phases: first replay reality, then infer from it.

1. Run the two baseline scenarios first.
   Start with `01_h100_baseline_pp2` and `02_a100_baseline_pp2` to measure clean H100-only and A100-only stage behavior.

2. Measure communication separately.
   Use `link_bench.py` to collect point-to-point and all-reduce timings for the tensor sizes that matter to your pipeline.

3. Keep the first pass in `explicit` mode.
   In this phase, HOPS is validating scheduling, topology, and heterogeneous imbalance while replaying measured stage times rather than trying to infer them.

4. Run the heterogeneous pair.
   Compare `03_hetero_even_pp4` against `04_hetero_weighted_pp4` and check whether the real cluster and HOPS agree on ranking, bottleneck stages, and the approximate gain from weighting the H100 stages more heavily.

5. Replace placeholders with measurements.
   Update the stage distributions in each `hops.yaml` from the Megatron logs, and update `overrides.links` from the communication benchmark.

6. Move to `analytical` mode only after the explicit replay is credible.
   The long-term goal is not to avoid measurement entirely, but to measure enough once that HOPS can predict new heterogeneous layouts with useful accuracy.

## Recommended Next Step After The First Pass

After the first baseline runs, replace the placeholder explicit stage distributions in the `hops.yaml` files with measured means and standard deviations from the Megatron logs. Then rerun the heterogeneous scenarios and compare ranking and error.
