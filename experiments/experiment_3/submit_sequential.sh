#!/bin/bash
# Submit experiment_3 scenarios as a single dependency chain.
# Set DRY_RUN=1 to print the plan without submitting. For dry runs, prefer
# MANIFEST=/tmp/exp3_sequential.tsv so checked-in Slurm results are not
# overwritten.

set -euo pipefail

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
EXPERIMENT_DIR="$REPO_DIR/experiments/experiment_3"
MANIFEST="${MANIFEST:-$EXPERIMENT_DIR/sequential_jobs.tsv}"
DRY_RUN="${DRY_RUN:-0}"

SCENARIOS=(
  "01_a10g_h100_pp2"
  "02_l4_h100_pp2"
  "03_l4_a10g_pp2"
  "04_h100_pair_batch7_pp2"
  "05_a10g_pair_batch7_pp2"
  "06_l4_pair_batch7_pp2"
  "07_h100_l4_seq1536_pp2"
  "08_a10g_l4_hidden1280_pp2"
  "09_h100_l4_a10g_pp3"
  "10_a10g_h100_l4_pp3"
  "11_a10g_l4_h100_pp3"
  "12_l4_h100_a10g_pp3"
  "13_mixed_pp3_batch7"
  "14_mixed_pp3_seq1536"
  "15_mixed_pp3_hidden1280"
  "16_mixed_pp3_layers15"
  "17_g_only_l4_a10g_l4_a10g_pp4"
  "18_g_only_a10g_l4_l4_a10g_pp4"
  "19_g_only_seq1536_pp4"
  "20_g_only_hidden1280_pp4"
  "21_g_only_layers16_pp4"
  "22_g_only_batch10_pp4"
  "23_mixed_h100_a10g_l4_h100_pp4"
  "24_mixed_l4_h100_a10g_h100_pp4"
  "25_pp5_h100_l4_a10g_l4_a10g"
  "26_pp5_l4_h100_a10g_l4_a10g"
  "27_pp5_a10g_h100_l4_a10g_l4"
  "28_pp5_batch10_h100_l4_a10g_a10g_l4"
  "29_pp5_seq1536_l4_a10g_h100_a10g_l4"
  "30_pp5_hidden1280_a10g_l4_h100_a10g_l4"
  "31_pp5_layers15_h100_a10g_l4_a10g_l4"
  "32_pp5_uneven_first5_last4"
  "33_all_nodes_order_h100_a10g_h100_l4_a10g_l4"
  "34_all_nodes_order_a10g_h100_l4_h100_l4_a10g"
  "35_all_nodes_batch1"
  "36_all_nodes_batch9"
  "37_all_nodes_batch15"
  "38_all_nodes_hidden1280"
  "39_all_nodes_seq1536"
  "40_all_nodes_hidden1536_seq768"
  "41_all_nodes_layers24"
  "42_all_nodes_uneven_first4_last6"
  "43_diag_all_nodes_gpipe_order_h100_a10g_h100_l4_a10g_l4"
  "44_diag_all_nodes_zero_bubble_batch9"
  "45_diag_g_only_gpipe_l4_a10g_l4_a10g_pp4"
  "46_diag_g_only_zero_bubble_batch10_pp4"
  "47_diag_pp3_gpipe_a10g_h100_l4"
  "48_diag_pp5_zero_bubble_h100_l4_a10g_l4_a10g"
)

submit_job() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN sbatch'
    printf " %q" "$@"
    printf '\n'
    SUBMITTED_JOB_ID="dryrun-${DRY_RUN_COUNTER}"
    DRY_RUN_COUNTER=$((DRY_RUN_COUNTER + 1))
    return 0
  fi

  SUBMITTED_JOB_ID=$(sbatch --parsable "$@")
}

echo -e "scenario	run_job_id	link_bench_job_id" > "$MANIFEST"

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
  echo -e "${scenario}	${run_job}	${link_job}" >> "$MANIFEST"
  previous_link_job="$link_job"
done

echo "wrote sequential manifest to $MANIFEST"
