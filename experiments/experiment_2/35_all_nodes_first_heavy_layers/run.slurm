#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_2/35_all_nodes_first_heavy_layers/run.slurm

#SBATCH --job-name=exp2-35-run
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

SCENARIO_REL="experiments/experiment_2/35_all_nodes_first_heavy_layers"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-30150}"
export PIPELINE_PARALLEL_SIZE=6
export NUM_LAYERS=18
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=4096
export NUM_ATTENTION_HEADS=8
export SEQ_LENGTH=1024
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=6
export TRAIN_ITERS=20
MEGATRON_EXTRA_ARGS="${MEGATRON_EXTRA_ARGS:-} --decoder-first-pipeline-num-layers 8 --decoder-last-pipeline-num-layers 2"
export MEGATRON_EXTRA_ARGS

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_megatron_job.sh"
