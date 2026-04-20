# Experiment 4 — HopsHetero validation pack for AWS ParallelCluster

Experiment 4 packages a small, reproducible PP4 matrix that targets the
heterogeneous scheduling finding from `HeteroResearch.md`:
`hops_hetero` should be Pareto-safe versus `zero_bubble` and improve makespan
when deferred W-gradients would otherwise spill into a cooldown tail.

This pack is intended for **self-serve validation on your own AWS
ParallelCluster** (Slurm), using the same g-hetero shape as experiment 3:
2×A10G + 2×L4, one GPU per node.

## Scenarios

- `01_pp4_l4_middle_mb24` — stage order `A10G, L4, L4, A10G`
- `02_pp4_l4_ends_mb24` — stage order `L4, A10G, A10G, L4`
- `03_pp4_l4_middle_mb48` — same order as #1, larger microbatch count
- `04_pp4_l4_ends_mb48` — same order as #2, larger microbatch count

Each scenario includes:

- `run.slurm` — Megatron + HOPS co-run (writes traces and HOPS summary)
- `link_bench.slurm` — pairwise communication calibration helper
- `scenario.env` — stage→GPU mapping for deterministic node assignment
- `hops.base.yaml` — 1F1B baseline (matches Megatron schedule)
- `hops.zero_bubble.yaml` — ZeroBubble analytical replay
- `hops.hops_hetero.yaml` — HopsHetero analytical replay

## Submit from head node

```bash
cd /home/ubuntu/HOPS
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export MEGATRON_ENTRYPOINT=pretrain_gpt.py
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
bash experiments/experiment_4/submit_sequential.sh
```

Dry run (writes a dependency plan only):

```bash
DRY_RUN=1 MANIFEST=/tmp/exp4_sequential.tsv   bash experiments/experiment_4/submit_sequential.sh
```

## Optional: scheduler-only HOPS comparison (no Megatron run)

```bash
bash experiments/experiment_4/compare_schedules.sh
```

This emits per-scenario JSON summaries in:
`experiments/experiment_4/<scenario>/output/hops_schedule_compare/`.

## Optional cluster template

A starter ParallelCluster config is provided at
`cluster_g5_g6_4gpu_template.yaml`. Replace the placeholder subnet IDs,
key name, and reservation IDs before use.
