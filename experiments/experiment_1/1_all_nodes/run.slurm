#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_1/1_all_nodes/run.slurm
#
# This script assumes four single-GPU G-family nodes in the g-hetero partition:
# two A10G nodes from g5.2xlarge and two L4 nodes from g6.4xlarge.

#SBATCH --job-name=exp1-1-all-g-nodes
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_1/1_all_nodes"
EXPECTED_NODE_COUNT=4
EXPECTED_GPU_PATTERN_BY_STAGE=("A10G" "A10G" "L4" "L4")
DEVICE_ID_BY_STAGE=(
  "a10g_node0_gpu0"
  "a10g_node1_gpu0"
  "l4_node0_gpu0"
  "l4_node1_gpu0"
)
STAGE_NODE_LABEL_BY_STAGE=(
  "a10g_node0"
  "a10g_node1"
  "l4_node0"
  "l4_node1"
)
MASTER_PORT="${MASTER_PORT:-29530}"
export PIPELINE_PARALLEL_SIZE=4
export NUM_LAYERS=8
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=4096
export NUM_ATTENTION_HEADS=8
export SEQ_LENGTH=1024
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=4
export TRAIN_ITERS=20

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_megatron_job.sh"
