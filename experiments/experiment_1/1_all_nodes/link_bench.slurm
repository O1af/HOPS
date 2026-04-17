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

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
SCENARIO_DIR="$REPO_DIR/experiments/experiment_1/1_all_nodes"
OUTPUT_DIR="$SCENARIO_DIR/output/${SLURM_JOB_ID:-manual}"
LINK_BENCH_DIR="$OUTPUT_DIR/calibration/link_bench"
mkdir -p "$LINK_BENCH_DIR"

TRAIN_ENV_ACTIVATE="${TRAIN_ENV_ACTIVATE:-}"
MASTER_PORT="${MASTER_PORT:-29540}"
SIZES_MB="${SIZES_MB:-1,2,4,16,64}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"

# Node ordering mirrors run.slurm: A10G nodes first as stages 0,1 then L4 nodes
# as stages 2,3. run_validation.py expects labels pair_0_1, pair_1_2, pair_2_3
# to match hops_trace_map.json.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#NODES[@]} -ne 4 ]]; then
  echo "Expected exactly 4 allocated nodes, got ${#NODES[@]}" >&2
  exit 1
fi

gpu_name_for_node() {
  local node=$1
  srun --nodes=1 --ntasks=1 -w "$node" bash -lc \
    'nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1'
}

A10G_NODES=()
L4_NODES=()
for node in "${NODES[@]}"; do
  gpu_name=$(gpu_name_for_node "$node")
  if [[ "$gpu_name" =~ A10G ]]; then
    A10G_NODES+=("$node")
  elif [[ "$gpu_name" =~ L4 ]]; then
    L4_NODES+=("$node")
  else
    echo "Unexpected GPU on $node: $gpu_name" >&2
    exit 1
  fi
done

if [[ ${#A10G_NODES[@]} -ne 2 || ${#L4_NODES[@]} -ne 2 ]]; then
  echo "Expected 2 A10G nodes and 2 L4 nodes." >&2
  exit 1
fi

ORDERED_NODES=(
  "${A10G_NODES[0]}"
  "${A10G_NODES[1]}"
  "${L4_NODES[0]}"
  "${L4_NODES[1]}"
)

{
  echo "stage0=${ORDERED_NODES[0]}"
  echo "stage1=${ORDERED_NODES[1]}"
  echo "stage2=${ORDERED_NODES[2]}"
  echo "stage3=${ORDERED_NODES[3]}"
  echo "link_bench_dir=$LINK_BENCH_DIR"
} > "$OUTPUT_DIR/node_inventory.txt"

cat > "$OUTPUT_DIR/launch_link_bench.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

NODE_RANK=$1

if [[ -n "${TRAIN_ENV_ACTIVATE:-}" ]]; then
  source "$TRAIN_ENV_ACTIVATE"
fi

cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_PER_NODE:-0}"
export NCCL_DEBUG=WARN
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

python -m torch.distributed.run \
  --nnodes=2 \
  --nproc_per_node=1 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$PAIR_MASTER_PORT" \
  experiments/link_bench.py \
    --mode p2p \
    --sizes-mb "$SIZES_MB" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --dtype "$DTYPE" \
    --label "$PAIR_LABEL"
EOF
chmod +x "$OUTPUT_DIR/launch_link_bench.sh"

export REPO_DIR SCENARIO_DIR TRAIN_ENV_ACTIVATE SIZES_MB DTYPE WARMUP ITERS

for pair in "0 1" "1 2" "2 3"; do
  read -r SRC DST <<< "$pair"
  SRC_NODE="${ORDERED_NODES[$SRC]}"
  DST_NODE="${ORDERED_NODES[$DST]}"
  PAIR_LABEL="pair_${SRC}_${DST}"
  PAIR_MASTER_PORT=$((MASTER_PORT + SRC))
  export MASTER_ADDR="$SRC_NODE" PAIR_MASTER_PORT PAIR_LABEL

  OUT_FILE="$LINK_BENCH_DIR/${PAIR_LABEL}.jsonl"
  : > "$OUT_FILE"

  PIDS=()
  for idx in 0 1; do
    if [[ "$idx" == "0" ]]; then
      node="$SRC_NODE"
    else
      node="$DST_NODE"
    fi
    srun --nodes=1 --ntasks=1 -w "$node" bash "$OUTPUT_DIR/launch_link_bench.sh" "$idx" \
      >> "$OUT_FILE" 2>> "$OUTPUT_DIR/${PAIR_LABEL}.err" &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
done

echo "wrote link bench output to $LINK_BENCH_DIR"
