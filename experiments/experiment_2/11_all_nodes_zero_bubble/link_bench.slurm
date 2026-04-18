#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2 ZeroBubble schedule scenario.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/11_all_nodes_zero_bubble/link_bench.slurm

#SBATCH --job-name=exp2-11-link-bench
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

SCENARIO_REL="experiments/experiment_2/11_all_nodes_zero_bubble"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-29770}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
