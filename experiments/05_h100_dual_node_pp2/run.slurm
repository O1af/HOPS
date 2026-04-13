#!/bin/bash
# Submit from the ParallelCluster head node:
#   sbatch experiments/05_h100_dual_node_pp2/run.slurm
#
# This script assumes the Slurm allocation contains exactly two H100 nodes and
# uses GPU 0 on each node.

#SBATCH --job-name=h100-dual-node-pp2
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
SCENARIO_DIR="$REPO_DIR/experiments/05_h100_dual_node_pp2"
OUTPUT_DIR="$SCENARIO_DIR/output/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUTPUT_DIR"

MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your shared Megatron-LM checkout}"
MEGATRON_ENTRYPOINT="${MEGATRON_ENTRYPOINT:-pretrain_gpt.py}"
TRAIN_ENV_ACTIVATE="${TRAIN_ENV_ACTIVATE:-}"
CUDA_VISIBLE_DEVICES_PER_NODE="${CUDA_VISIBLE_DEVICES_PER_NODE:-0}"
MASTER_PORT="${MASTER_PORT:-29520}"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES_PER_NODE"
LOCAL_GPU_COUNT=${#GPU_IDS[@]}

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [[ ${#NODES[@]} -ne 2 ]]; then
  echo "Expected exactly 2 allocated nodes, got ${#NODES[@]}" >&2
  exit 1
fi

gpu_name_for_node() {
  local node=$1
  srun --nodes=1 --ntasks=1 -w "$node" bash -lc \
    'nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1'
}

NODE0_GPU=$(gpu_name_for_node "${NODES[0]}")
NODE1_GPU=$(gpu_name_for_node "${NODES[1]}")

if [[ ! "$NODE0_GPU" =~ H100 || ! "$NODE1_GPU" =~ H100 ]]; then
  echo "Expected two H100 nodes:" >&2
  echo "  ${NODES[0]} -> $NODE0_GPU" >&2
  echo "  ${NODES[1]} -> $NODE1_GPU" >&2
  exit 1
fi

{
  echo "node0=${NODES[0]} gpu=$NODE0_GPU"
  echo "node1=${NODES[1]} gpu=$NODE1_GPU"
  echo "cuda_visible_devices_per_node=$CUDA_VISIBLE_DEVICES_PER_NODE"
} > "$OUTPUT_DIR/node_inventory.txt"

cat > "$OUTPUT_DIR/launch_megatron.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

NODE_RANK=$1

if [[ -n "${TRAIN_ENV_ACTIVATE:-}" ]]; then
  source "$TRAIN_ENV_ACTIVATE"
fi

cd "$MEGATRON_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_PER_NODE"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

MEGATRON_ARGS=(
  "$MEGATRON_ENTRYPOINT"
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --num-layers "$NUM_LAYERS"
  --hidden-size "$HIDDEN_SIZE"
  --ffn-hidden-size "$FFN_HIDDEN_SIZE"
  --num-attention-heads "$NUM_ATTENTION_HEADS"
  --seq-length "$SEQ_LENGTH"
  --max-position-embeddings "$SEQ_LENGTH"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --train-iters "$TRAIN_ITERS"
  --eval-interval "$TRAIN_ITERS"
  --eval-iters 0
  --lr 1.0e-4
  --min-lr 1.0e-4
  --lr-decay-style constant
  --bf16
  --transformer-impl local
  --no-persist-layer-norm
  --no-gradient-accumulation-fusion
  --no-masked-softmax-fusion
  --mock-data
  --tokenizer-type NullTokenizer
  --vocab-size 50304
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --log-interval 1
  --log-throughput
)

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$LOCAL_GPU_COUNT" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "${MEGATRON_ARGS[@]}"
EOF
chmod +x "$OUTPUT_DIR/launch_megatron.sh"

cd "$REPO_DIR"
env -u VIRTUAL_ENV uv run python main.py \
  --config "$SCENARIO_DIR/hops.yaml" \
  --no-viz \
  --summary-json "$OUTPUT_DIR/hops_summary.json" \
  --trace-csv "$OUTPUT_DIR/hops_trace.csv" \
  > "$OUTPUT_DIR/hops_stdout.txt"

export MEGATRON_DIR
export MEGATRON_ENTRYPOINT
export TRAIN_ENV_ACTIVATE
export CUDA_VISIBLE_DEVICES_PER_NODE
export MASTER_ADDR="${NODES[0]}"
export MASTER_PORT
export NNODES=2
export LOCAL_GPU_COUNT
export PIPELINE_PARALLEL_SIZE=2
export NUM_LAYERS=8
export HIDDEN_SIZE=3072
export FFN_HIDDEN_SIZE=12288
export NUM_ATTENTION_HEADS=24
export SEQ_LENGTH=2048
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=8
export TRAIN_ITERS=40

srun --nodes=1 --ntasks=1 -w "${NODES[0]}" bash "$OUTPUT_DIR/launch_megatron.sh" 0 \
  > "$OUTPUT_DIR/megatron_node0.log" 2>&1 &
PID0=$!

srun --nodes=1 --ntasks=1 -w "${NODES[1]}" bash "$OUTPUT_DIR/launch_megatron.sh" 1 \
  > "$OUTPUT_DIR/megatron_node1.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"
