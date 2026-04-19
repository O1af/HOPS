#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_2/27_g_only_alternating_pp4/run.slurm

#SBATCH --job-name=exp2-27-run
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_2/27_g_only_alternating_pp4"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-30070}"
export PIPELINE_PARALLEL_SIZE=4
export NUM_LAYERS=12
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=4096
export NUM_ATTENTION_HEADS=8
export SEQ_LENGTH=1024
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=4
export TRAIN_ITERS=20
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_megatron_job.sh"
