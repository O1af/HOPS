#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2 H100 PP2.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/8_h100_pair_pp2/link_bench.slurm

#SBATCH --job-name=exp2-8-link-bench
#SBATCH --partition=p5-cap
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_2/8_h100_pair_pp2"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-29710}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
