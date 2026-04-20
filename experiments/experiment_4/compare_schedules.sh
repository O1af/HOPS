#!/bin/bash
# Run HOPS-only schedule comparison on experiment_4 configs.
set -euo pipefail

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
EXPERIMENT_DIR="$REPO_DIR/experiments/experiment_4"

SCENARIOS=(
  "01_pp4_l4_middle_mb24"
  "02_pp4_l4_ends_mb24"
  "03_pp4_l4_middle_mb48"
  "04_pp4_l4_ends_mb48"
)
CONFIGS=(
  "hops.base.yaml"
  "hops.zero_bubble.yaml"
  "hops.hops_hetero.yaml"
)

for scenario in "${SCENARIOS[@]}"; do
  scenario_dir="$EXPERIMENT_DIR/$scenario"
  out_dir="$scenario_dir/output/hops_schedule_compare"
  mkdir -p "$out_dir"

  for cfg in "${CONFIGS[@]}"; do
    cfg_path="$scenario_dir/$cfg"
    tag="${cfg%.yaml}"
    env -u VIRTUAL_ENV uv run python "$REPO_DIR/main.py" \
      --config "$cfg_path" \
      --no-viz \
      --summary-json "$out_dir/${tag}.summary.json" \
      --trace-csv "$out_dir/${tag}.trace.csv" \
      > "$out_dir/${tag}.stdout.txt"
  done
done

echo "wrote schedule-comparison artifacts under experiments/experiment_4/*/output/hops_schedule_compare"
