#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2 h100_pair_gpipe.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/22_h100_pair_gpipe/link_bench.slurm

#SBATCH --job-name=exp2-22-link
#SBATCH --partition=p5-cap
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_2/22_h100_pair_gpipe"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31020}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
