#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_3/20_g_only_hidden1280_pp4/link.slurm

#SBATCH --job-name=exp3-20-link
#SBATCH --output=%x-%j.out
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive

set -euo pipefail

SCENARIO_REL="experiments/experiment_3/20_g_only_hidden1280_pp4"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31220}"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
