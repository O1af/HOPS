#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_3/40_all_nodes_hidden1536_seq768/run.slurm

#SBATCH --job-name=exp3-40-run
#SBATCH --output=%x-%j.out
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH hetjob
#SBATCH --partition=p5-cap
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -euo pipefail

SCENARIO_REL="experiments/experiment_3/40_all_nodes_hidden1536_seq768"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-30240}"
export PIPELINE_PARALLEL_SIZE=6
export NUM_LAYERS=12
export HIDDEN_SIZE=1536
export FFN_HIDDEN_SIZE=6144
export NUM_ATTENTION_HEADS=12
export SEQ_LENGTH=768
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=6
export TRAIN_ITERS=20
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_megatron_job.sh"
