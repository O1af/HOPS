#!/bin/bash
# Submit experiment_2 scenarios as a single dependency chain.
# Set DRY_RUN=1 to print the plan without submitting.

set -euo pipefail

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
EXPERIMENT_DIR="$REPO_DIR/experiments/experiment_2"
MANIFEST="${MANIFEST:-$EXPERIMENT_DIR/sequential_jobs.tsv}"
DRY_RUN="${DRY_RUN:-0}"

SCENARIOS=(
  "1_all_nodes"
  "10_all_nodes_gpipe"
  "11_all_nodes_zero_bubble"
  "8_h100_pair_pp2"
  "9_g_only_pp4"
  "12_g_only_zero_bubble"
  "2_reverse_order"
  "3_interleaved"
  "4_slow_middle"
  "5_small_model"
  "7_batch12"
  "6_large_model"
)

submit_job() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN sbatch'
    printf ' %q' "$@"
    printf '\n'
    SUBMITTED_JOB_ID="dryrun-${DRY_RUN_COUNTER}"
    DRY_RUN_COUNTER=$((DRY_RUN_COUNTER + 1))
    return 0
  fi

  SUBMITTED_JOB_ID=$(sbatch --parsable "$@")
}

echo -e "scenario\trun_job_id\tlink_bench_job_id" > "$MANIFEST"

previous_link_job=""
DRY_RUN_COUNTER=1
for scenario in "${SCENARIOS[@]}"; do
  scenario_dir="$EXPERIMENT_DIR/$scenario"
  run_script="$scenario_dir/run.slurm"
  link_script="$scenario_dir/link_bench.slurm"

  if [[ ! -f "$run_script" || ! -f "$link_script" ]]; then
    echo "Missing run or link-bench script for $scenario" >&2
    exit 1
  fi

  run_args=()
  if [[ -n "$previous_link_job" ]]; then
    run_args+=("--dependency=afterok:$previous_link_job")
  fi
  run_args+=("$run_script")
  submit_job "${run_args[@]}"
  run_job="$SUBMITTED_JOB_ID"

  submit_job "--dependency=afterok:$run_job" "$link_script"
  link_job="$SUBMITTED_JOB_ID"
  echo -e "${scenario}\t${run_job}\t${link_job}" >> "$MANIFEST"
  previous_link_job="$link_job"
done

echo "wrote sequential manifest to $MANIFEST"
