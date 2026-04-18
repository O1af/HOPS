#!/bin/bash

collect_allocated_nodes() {
  local found_het_group=0
  local idx var_name nodelist

  for idx in {0..31}; do
    var_name="SLURM_JOB_NODELIST_HET_GROUP_${idx}"
    nodelist="${!var_name:-}"
    if [[ -n "$nodelist" ]]; then
      found_het_group=1
      scontrol show hostnames "$nodelist"
    fi
  done

  if [[ "$found_het_group" -eq 0 ]]; then
    scontrol show hostnames "$SLURM_JOB_NODELIST"
  fi
}

gpu_name_for_node() {
  local node=$1
  srun --nodes=1 --ntasks=1 -w "$node" bash -lc \
    'nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1'
}

is_node_used() {
  local needle=$1
  local node

  for node in "${ORDERED_NODES[@]:-}"; do
    if [[ "$node" == "$needle" ]]; then
      return 0
    fi
  done

  return 1
}

assign_ordered_nodes() {
  local stage_count=${#EXPECTED_GPU_PATTERN_BY_STAGE[@]}
  local device_count=${#DEVICE_ID_BY_STAGE[@]}
  local label_count=${#STAGE_NODE_LABEL_BY_STAGE[@]}
  local node gpu_name stage pattern selected

  if [[ "$stage_count" -eq 0 ]]; then
    echo "EXPECTED_GPU_PATTERN_BY_STAGE must contain at least one stage" >&2
    exit 1
  fi
  if [[ "$device_count" -ne "$stage_count" ]]; then
    echo "DEVICE_ID_BY_STAGE has $device_count entries, expected $stage_count" >&2
    exit 1
  fi
  if [[ "$label_count" -ne "$stage_count" ]]; then
    echo "STAGE_NODE_LABEL_BY_STAGE has $label_count entries, expected $stage_count" >&2
    exit 1
  fi
  if [[ "${EXPECTED_NODE_COUNT:-$stage_count}" -ne "$stage_count" ]]; then
    echo "EXPECTED_NODE_COUNT=${EXPECTED_NODE_COUNT:-unset} does not match stage count $stage_count" >&2
    exit 1
  fi

  NODES=()
  while IFS= read -r node; do
    NODES+=("$node")
  done < <(collect_allocated_nodes)
  if [[ ${#NODES[@]} -ne "$stage_count" ]]; then
    echo "Expected exactly $stage_count allocated nodes, got ${#NODES[@]}" >&2
    printf 'nodes: %s\n' "${NODES[@]}" >&2
    env | sort | grep -E 'SLURM_.*NODELIST|SLURM_HET' >&2 || true
    exit 1
  fi

  ORDERED_NODES=()
  NODE_GPU_NAMES=()
  for node in "${NODES[@]}"; do
    gpu_name=$(gpu_name_for_node "$node")
    NODE_GPU_NAMES+=("$node=$gpu_name")
  done

  for ((stage = 0; stage < stage_count; stage++)); do
    pattern="${EXPECTED_GPU_PATTERN_BY_STAGE[$stage]}"
    selected=""
    for node in "${NODES[@]}"; do
      if is_node_used "$node"; then
        continue
      fi
      gpu_name=""
      for entry in "${NODE_GPU_NAMES[@]}"; do
        if [[ "$entry" == "$node="* ]]; then
          gpu_name="${entry#*=}"
          break
        fi
      done
      if [[ "$gpu_name" =~ $pattern ]]; then
        selected="$node"
        break
      fi
    done

    if [[ -z "$selected" ]]; then
      echo "Could not find unused node matching GPU pattern '$pattern' for stage $stage" >&2
      printf 'GPU inventory:\n' >&2
      printf '  %s\n' "${NODE_GPU_NAMES[@]}" >&2
      exit 1
    fi

    ORDERED_NODES+=("$selected")
  done
}

write_trace_map() {
  local output_file=$1
  local stage last_stage suffix

  last_stage=$((${#DEVICE_ID_BY_STAGE[@]} - 1))
  {
    echo "{"
    echo '  "stages": {'
    for stage in "${!DEVICE_ID_BY_STAGE[@]}"; do
      suffix=","
      if [[ "$stage" -eq "$last_stage" ]]; then
        suffix=""
      fi
      printf '    "%s": "%s"%s\n' "$stage" "${DEVICE_ID_BY_STAGE[$stage]}" "$suffix"
    done
    echo "  }"
    echo "}"
  } > "$output_file"
}

write_run_node_inventory() {
  local output_file=$1
  local stage

  {
    for stage in "${!ORDERED_NODES[@]}"; do
      echo "${STAGE_NODE_LABEL_BY_STAGE[$stage]}=${ORDERED_NODES[$stage]}"
    done
    echo "ordered_nodes=${ORDERED_NODES[*]}"
    echo "cuda_visible_devices_per_node=$CUDA_VISIBLE_DEVICES_PER_NODE"
    echo "trace_dir=$TRACE_DIR"
    echo "torch_profiler_dir=$PROFILER_DIR"
    echo "trace_map_file=$TRACE_MAP_FILE"
  } > "$output_file"
}

write_link_bench_node_inventory() {
  local output_file=$1
  local stage

  {
    for stage in "${!ORDERED_NODES[@]}"; do
      echo "stage${stage}=${ORDERED_NODES[$stage]}"
    done
    echo "link_bench_dir=$LINK_BENCH_DIR"
  } > "$output_file"
}

require_model_env() {
  local missing=()
  local name

  for name in \
    PIPELINE_PARALLEL_SIZE \
    NUM_LAYERS \
    HIDDEN_SIZE \
    FFN_HIDDEN_SIZE \
    NUM_ATTENTION_HEADS \
    SEQ_LENGTH \
    MICRO_BATCH_SIZE \
    GLOBAL_BATCH_SIZE \
    TRAIN_ITERS
  do
    if [[ -z "${!name:-}" ]]; then
      missing+=("$name")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Missing required Megatron model setting(s): ${missing[*]}" >&2
    exit 1
  fi
}

validate_single_gpu_per_node() {
  IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES_PER_NODE"
  LOCAL_GPU_COUNT=${#GPU_IDS[@]}
  if [[ "$LOCAL_GPU_COUNT" -ne "${EXPECTED_LOCAL_GPU_COUNT:-1}" ]]; then
    echo "Expected exactly ${EXPECTED_LOCAL_GPU_COUNT:-1} visible GPU per node, got $LOCAL_GPU_COUNT from CUDA_VISIBLE_DEVICES_PER_NODE=$CUDA_VISIBLE_DEVICES_PER_NODE" >&2
    exit 1
  fi
}
