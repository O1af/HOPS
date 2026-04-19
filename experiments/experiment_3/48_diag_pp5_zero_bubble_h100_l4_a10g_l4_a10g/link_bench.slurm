#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_3/48_diag_pp5_zero_bubble_h100_l4_a10g_l4_a10g/link.slurm

#SBATCH --job-name=exp3-48-link
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

SCENARIO_REL="experiments/experiment_3/48_diag_pp5_zero_bubble_h100_l4_a10g_l4_a10g"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31248}"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
