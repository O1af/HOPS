#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for experiment_2 mixed_pp5_g4_h100_tail.
# Submit after run.slurm completes:
#   sbatch experiments/experiment_2/21_mixed_pp5_g4_h100_tail/link_bench.slurm

#SBATCH --job-name=exp2-21-link
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

SCENARIO_REL="experiments/experiment_2/21_mixed_pp5_g4_h100_tail"
source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/$SCENARIO_REL/scenario.env"
MASTER_PORT="${MASTER_PORT:-31010}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
