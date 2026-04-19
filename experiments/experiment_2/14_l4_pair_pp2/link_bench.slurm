#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2 l4_pair_pp2.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/14_l4_pair_pp2/link_bench.slurm

#SBATCH --job-name=exp2-14-link
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_2/14_l4_pair_pp2"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-30940}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
