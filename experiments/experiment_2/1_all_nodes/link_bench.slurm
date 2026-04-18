#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/1_all_nodes/link_bench.slurm

#SBATCH --job-name=exp2-1-all-link-bench
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out
#SBATCH hetjob
#SBATCH --partition=p5-cap
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -euo pipefail

SCENARIO_REL="experiments/experiment_2/1_all_nodes"
EXPECTED_NODE_COUNT=6
EXPECTED_GPU_PATTERN_BY_STAGE=("H100" "H100" "A10G" "A10G" "L4" "L4")
DEVICE_ID_BY_STAGE=(
  "h100_node0_gpu0"
  "h100_node1_gpu0"
  "a10g_node0_gpu0"
  "a10g_node1_gpu0"
  "l4_node0_gpu0"
  "l4_node1_gpu0"
)
STAGE_NODE_LABEL_BY_STAGE=(
  "h100_node0"
  "h100_node1"
  "a10g_node0"
  "a10g_node1"
  "l4_node0"
  "l4_node1"
)
MASTER_PORT="${MASTER_PORT:-29570}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
