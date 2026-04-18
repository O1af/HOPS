#!/bin/bash
# Measure adjacent-pipeline-pair p2p bandwidth and latency for the four-node
# G-family pipeline defined by this scenario. Submit AFTER a training run.slurm
# job has landed so its node inventory and trace map can be matched up.
#
#   sbatch experiments/experiment_1/1_all_nodes/link_bench.slurm
#
# Writes JSONL under $SCENARIO_DIR/output/<this-job-id>/calibration/link_bench/
# so you can point run_validation.py at it via --link-bench-dir.

#SBATCH --job-name=exp1-1-all-link-bench
#SBATCH --partition=g-hetero
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_REL="experiments/experiment_1/1_all_nodes"
EXPECTED_NODE_COUNT=4
EXPECTED_GPU_PATTERN_BY_STAGE=("A10G" "A10G" "L4" "L4")
DEVICE_ID_BY_STAGE=(
  "a10g_node0_gpu0"
  "a10g_node1_gpu0"
  "l4_node0_gpu0"
  "l4_node1_gpu0"
)
STAGE_NODE_LABEL_BY_STAGE=(
  "a10g_node0"
  "a10g_node1"
  "l4_node0"
  "l4_node1"
)
MASTER_PORT="${MASTER_PORT:-29540}"

source "$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)/experiments/lib/run_link_bench_job.sh"
