#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/experiment_3/04_h100_pair_batch7_pp2/link.slurm

#SBATCH --job-name=exp3-04-link
#SBATCH --output=%x-%j.out
#SBATCH --partition=p5-cap
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

set -euo pipefail

SCENARIO_REL="experiments/experiment_3/04_h100_pair_batch7_pp2"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31204}"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
