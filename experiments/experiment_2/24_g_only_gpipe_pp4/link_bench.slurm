#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2 g_only_gpipe_pp4.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/24_g_only_gpipe_pp4/link_bench.slurm

#SBATCH --job-name=exp2-24-link
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_2/24_g_only_gpipe_pp4"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31040}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
