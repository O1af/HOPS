#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_3/37_all_nodes_batch15/link.slurm

#SBATCH --job-name=exp3-37-link
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

SCENARIO_REL="experiments/experiment_3/37_all_nodes_batch15"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31237}"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
