#!/bin/bash
# Submit from the ParallelCluster head node, typically:
#   sbatch -p p5 experiments/01_h100_baseline_pp2/run.slurm

#SBATCH --job-name=h100-baseline-pp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --output=%x-%j.out

set -euo pipefail

SCENARIO_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCENARIO_DIR/../.." && pwd)
OUTPUT_DIR="$SCENARIO_DIR/output/${SLURM_JOB_ID:-manual}"
mkdir -p "$OUTPUT_DIR"

MEGATRON_DIR="${MEGATRON_DIR:?Set MEGATRON_DIR to your shared Megatron-LM checkout}"
MEGATRON_ENTRYPOINT="${MEGATRON_ENTRYPOINT:-pretrain_gpt.py}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29500}"

IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
LOCAL_GPU_COUNT=${#GPU_IDS[@]}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
if [[ ! "$GPU_NAME" =~ H100 ]]; then
  echo "Expected an H100 node, but found: $GPU_NAME" >&2
  exit 1
fi

{
  echo "hostname=$(hostname)"
  echo "gpu_name=$GPU_NAME"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
} > "$OUTPUT_DIR/node_inventory.txt"

cat > "$OUTPUT_DIR/launch_megatron.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

cd "$MEGATRON_DIR"

export CUDA_VISIBLE_DEVICES
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
  --lr 1.0e-4
  --min-lr 1.0e-4
  --lr-decay-style constant
  --bf16
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
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "${MEGATRON_ARGS[@]}"
EOF
chmod +x "$OUTPUT_DIR/launch_megatron.sh"

cd "$REPO_DIR"
uv run python main.py \
  --config "$SCENARIO_DIR/hops.yaml" \
  --no-viz \
  --summary-json "$OUTPUT_DIR/hops_summary.json" \
  --trace-csv "$OUTPUT_DIR/hops_trace.csv" \
  > "$OUTPUT_DIR/hops_stdout.txt"

export MEGATRON_DIR
export MEGATRON_ENTRYPOINT
export CUDA_VISIBLE_DEVICES
export MASTER_ADDR=127.0.0.1
export MASTER_PORT
export NNODES=1
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

bash "$OUTPUT_DIR/launch_megatron.sh" > "$OUTPUT_DIR/megatron.log" 2>&1
