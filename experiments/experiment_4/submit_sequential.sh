#!/bin/bash
# Submit experiment_4 scenarios as a dependency chain.
set -euo pipefail

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
EXPERIMENT_DIR="$REPO_DIR/experiments/experiment_4"
MANIFEST="${MANIFEST:-$EXPERIMENT_DIR/sequential_jobs.tsv}"
DRY_RUN="${DRY_RUN:-0}"

SCENARIOS=(
  "01_pp4_l4_middle_mb24"
  "02_pp4_l4_ends_mb24"
  "03_pp4_l4_middle_mb48"
  "04_pp4_l4_ends_mb48"
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
