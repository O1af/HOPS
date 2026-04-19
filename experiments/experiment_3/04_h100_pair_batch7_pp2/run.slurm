#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_3/04_h100_pair_batch7_pp2/run.slurm

#SBATCH --job-name=exp3-04-run
#SBATCH --output=%x-%j.out
#SBATCH --partition=p5-cap
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -euo pipefail

SCENARIO_REL="experiments/experiment_3/04_h100_pair_batch7_pp2"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-30204}"
export PIPELINE_PARALLEL_SIZE=2
export NUM_LAYERS=12
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=4096
export NUM_ATTENTION_HEADS=8
export SEQ_LENGTH=1024
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=7
export TRAIN_ITERS=20
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_megatron_job.sh"
