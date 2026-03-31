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

## Recommended Next Step After The First Pass

After the first baseline runs, replace the placeholder explicit stage distributions in the `hops.yaml` files with measured means and standard deviations from the Megatron logs. Then rerun the heterogeneous scenarios and compare ranking and error.
