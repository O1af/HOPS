#!/bin/bash

COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$COMMON_DIR/slurm_common.sh"

REPO_DIR=$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)
SCENARIO_DIR="$REPO_DIR/$SCENARIO_REL"
OUTPUT_DIR="$SCENARIO_DIR/output/${SLURM_JOB_ID:-manual}"
TRACE_DIR="$OUTPUT_DIR/megatron_trace"
PROFILER_DIR="$OUTPUT_DIR/torch_profiler"
TRACE_MAP_FILE="$OUTPUT_DIR/hops_trace_map.json"
mkdir -p "$OUTPUT_DIR" "$TRACE_DIR" "$PROFILER_DIR"

MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your shared Megatron-LM checkout}"
MEGATRON_ENTRYPOINT="${MEGATRON_ENTRYPOINT:-pretrain_gpt.py}"
TRAIN_ENV_ACTIVATE="${TRAIN_ENV_ACTIVATE:-}"
CUDA_VISIBLE_DEVICES_PER_NODE="${CUDA_VISIBLE_DEVICES_PER_NODE:-0}"
MASTER_PORT="${MASTER_PORT:-29520}"
MEGATRON_EXTRA_ARGS="${MEGATRON_EXTRA_ARGS:-}"
HOPS_TRACE_ENABLED="${HOPS_TRACE_ENABLED:-1}"
HOPS_TRACE_FORMAT="${HOPS_TRACE_FORMAT:-jsonl}"
HOPS_TRACE_START_ITER="${HOPS_TRACE_START_ITER:-3}"
HOPS_TRACE_END_ITER="${HOPS_TRACE_END_ITER:-0}"
HOPS_TRACE_WITH_TORCH_PROFILER="${HOPS_TRACE_WITH_TORCH_PROFILER:-0}"

validate_single_gpu_per_node
assign_ordered_nodes
write_trace_map "$TRACE_MAP_FILE"
write_run_node_inventory "$OUTPUT_DIR/node_inventory.txt"
require_model_env

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
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export HOPS_TRACE_ENABLED
export HOPS_TRACE_DIR
export HOPS_TRACE_FORMAT
export HOPS_TRACE_START_ITER
export HOPS_TRACE_END_ITER
export HOPS_TRACE_WITH_TORCH_PROFILER
export HOPS_TORCH_PROFILER_DIR
export HOPS_TRACE_MAP_FILE

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

if [[ -n "${MEGATRON_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( $MEGATRON_EXTRA_ARGS )
  MEGATRON_ARGS+=("${EXTRA_ARGS[@]}")
fi

python -m torch.distributed.run \
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
  --config "$SCENARIO_DIR/hops.base.yaml" \
  --no-viz \
  --summary-json "$OUTPUT_DIR/hops_summary.json" \
  --trace-csv "$OUTPUT_DIR/hops_trace.csv" \
  > "$OUTPUT_DIR/hops_stdout.txt"

export MEGATRON_DIR
export MEGATRON_ENTRYPOINT
export TRAIN_ENV_ACTIVATE
export CUDA_VISIBLE_DEVICES_PER_NODE
export MASTER_ADDR="${ORDERED_NODES[0]}"
export MASTER_PORT
export NNODES="${#ORDERED_NODES[@]}"
export LOCAL_GPU_COUNT
export MEGATRON_EXTRA_ARGS
export HOPS_TRACE_ENABLED
export HOPS_TRACE_FORMAT
export HOPS_TRACE_START_ITER
export HOPS_TRACE_END_ITER
export HOPS_TRACE_WITH_TORCH_PROFILER
export HOPS_TRACE_DIR="$TRACE_DIR"
export HOPS_TORCH_PROFILER_DIR="$PROFILER_DIR"
export HOPS_TRACE_MAP_FILE="$TRACE_MAP_FILE"
export PIPELINE_PARALLEL_SIZE
export NUM_LAYERS
export HIDDEN_SIZE
export FFN_HIDDEN_SIZE
export NUM_ATTENTION_HEADS
export SEQ_LENGTH
export MICRO_BATCH_SIZE
export GLOBAL_BATCH_SIZE
export TRAIN_ITERS

PIDS=()
for rank in "${!ORDERED_NODES[@]}"; do
  node=${ORDERED_NODES[$rank]}
  srun --nodes=1 --ntasks=1 -w "$node" bash "$OUTPUT_DIR/launch_megatron.sh" "$rank" \
    > "$OUTPUT_DIR/megatron_rank${rank}_${node}.log" 2>&1 &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

if compgen -G "$TRACE_DIR/*.jsonl" > /dev/null; then
  env -u VIRTUAL_ENV uv run python -m hops.megatron_cli --job-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/megatron_convert_stdout.txt"
fi
