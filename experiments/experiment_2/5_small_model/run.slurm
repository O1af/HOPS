#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_2/5_small_model/run.slurm

#SBATCH --job-name=exp2-5-small
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

SCENARIO_REL="experiments/experiment_2/5_small_model"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-29640}"
export PIPELINE_PARALLEL_SIZE=6
export NUM_LAYERS=6
export HIDDEN_SIZE=768
export FFN_HIDDEN_SIZE=3072
export NUM_ATTENTION_HEADS=6
export SEQ_LENGTH=512
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=6
export TRAIN_ITERS=30

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_megatron_job.sh"
