#!/bin/bash

COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$COMMON_DIR/slurm_common.sh"

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
SCENARIO_DIR="$REPO_DIR/$SCENARIO_REL"
OUTPUT_DIR="$SCENARIO_DIR/output/${SLURM_JOB_ID:-manual}"
LINK_BENCH_DIR="$OUTPUT_DIR/calibration/link_bench"
mkdir -p "$LINK_BENCH_DIR"

TRAIN_ENV_ACTIVATE="${TRAIN_ENV_ACTIVATE:-}"
CUDA_VISIBLE_DEVICES_PER_NODE="${CUDA_VISIBLE_DEVICES_PER_NODE:-0}"
MASTER_PORT="${MASTER_PORT:-29540}"
SIZES_MB="${SIZES_MB:-1,2,4,16,64}"
DTYPE="${DTYPE:-bfloat16}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-50}"

assign_ordered_nodes
write_link_bench_node_inventory "$OUTPUT_DIR/node_inventory.txt"

cat > "$OUTPUT_DIR/launch_link_bench.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

NODE_RANK=$1

if [[ -n "${TRAIN_ENV_ACTIVATE:-}" ]]; then
  source "$TRAIN_ENV_ACTIVATE"
fi

cd "$REPO_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_PER_NODE"
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

export REPO_DIR
export SCENARIO_DIR
export TRAIN_ENV_ACTIVATE
export CUDA_VISIBLE_DEVICES_PER_NODE
export SIZES_MB
export DTYPE
export WARMUP
export ITERS

for ((src = 0; src < ${#ORDERED_NODES[@]} - 1; src++)); do
  dst=$((src + 1))
  src_node="${ORDERED_NODES[$src]}"
  dst_node="${ORDERED_NODES[$dst]}"
  pair_label="pair_${src}_${dst}"
  pair_master_port=$((MASTER_PORT + src))
  export MASTER_ADDR="$src_node"
  export PAIR_MASTER_PORT="$pair_master_port"
  export PAIR_LABEL="$pair_label"

  out_file="$LINK_BENCH_DIR/${pair_label}.jsonl"
  : > "$out_file"

  PIDS=()
  for idx in 0 1; do
    if [[ "$idx" == "0" ]]; then
      node="$src_node"
    else
      node="$dst_node"
    fi
    srun --nodes=1 --ntasks=1 -w "$node" bash "$OUTPUT_DIR/launch_link_bench.sh" "$idx" \
      >> "$out_file" 2>> "$OUTPUT_DIR/${pair_label}.err" &
    PIDS+=("$!")
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
done

echo "wrote link bench output to $LINK_BENCH_DIR"
