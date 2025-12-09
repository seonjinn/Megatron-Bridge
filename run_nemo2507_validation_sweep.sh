#!/bin/bash
#
# NeMo 25.07 Reference Configuration Validation Sweep
# This script validates Megatron-Bridge against NeMo 25.07 reference configurations
#
# Usage: ./run_nemo2507_validation_sweep.sh [--dry-run] [--model MODEL_NAME]
#
# Options:
#   --dry-run       Print commands without executing
#   --model NAME    Run only specific model (llama3_8b, llama3_70b, llama31_405b, deepseek_v3, qwen3_30b, qwen3_235b, gpt_oss_20b)
#

set -e

# Get script directory for finding config files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Parse Arguments
# ============================================================================
DRY_RUN=false
FILTER_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            echo "=== DRY RUN MODE ==="
            shift
            ;;
        --model)
            FILTER_MODEL="$2"
            echo "=== Filtering to model: ${FILTER_MODEL} ==="
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--model MODEL_NAME]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Common Settings for GB200
# ============================================================================
ACCOUNT="coreai_dlalgo_nemorl"
# ACCOUNT="coreai_dlalgo_llm"

PARTITION="batch_long"
TIME_LIMIT="08:00:00"
CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"

# GB200 specific settings
GPU_TYPE="gb200"
GPUS_PER_NODE=4  # GB200 has 4 GPUs per node

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# HuggingFace settings
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Bridge-NeMo2507-Validation"
TENSORBOARD_DIR="./exp_logs/tensorboard"
MAX_STEPS=50  # NeMo 25.07 reference: 50 steps, average steps 35-50 (to match Malay's metric calculation)

# Results tracking
SWEEP_ID="nemo2507_validation_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="./exp_logs/sweeps/${SWEEP_ID}"
JOBS_FILE="${SWEEP_DIR}/submitted_jobs.txt"

mkdir -p "${SWEEP_DIR}"

# ============================================================================
# NeMo 25.07 Reference Configurations
# ============================================================================
# Configs are loaded from: gb200_reference_configs.yaml
# This is the single source of truth used by both:
#   - This sweep script (job launching)
#   - collect_results.py (result filtering and comparison)
#
# Format: "MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG"

CONFIG_FILE="${SCRIPT_DIR}/gb200_reference_configs.yaml"
CONFIG_SECTION="pretrain_bf16"  # Default section

# Check if config file exists
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    echo "Please ensure gb200_reference_configs.yaml exists in the same directory."
    exit 1
fi

# Function to load configs from YAML
load_configs_from_yaml() {
    local section="$1"
    local model_filter="$2"
    
    if [[ -n "$model_filter" ]]; then
        python3 scripts/parse_reference_config.py \
            --config "${CONFIG_FILE}" \
            --section "${section}" \
            --model "${model_filter}" \
            --format sweep 2>/dev/null
    else
        python3 scripts/parse_reference_config.py \
            --config "${CONFIG_FILE}" \
            --section "${section}" \
            --all \
            --format sweep 2>/dev/null
    fi
}

# Load all pretrain_bf16 configs into arrays
mapfile -t LLAMA3_8B_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "llama3_8b")
mapfile -t LLAMA3_70B_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "llama3_70b")
mapfile -t LLAMA31_405B_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "llama31_405b")
mapfile -t DEEPSEEK_V3_LARGE_GBS_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "deepseek_v3")
mapfile -t QWEN3_30B_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "qwen3_30b")
mapfile -t QWEN3_235B_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "qwen3_235b")
mapfile -t GPT_OSS_20B_2507_CONFIGS < <(load_configs_from_yaml "${CONFIG_SECTION}" "gpt_oss_20b")

# ============================================================================
# Build Experiment List Based on Filter
# ============================================================================
EXPERIMENTS=()

add_configs() {
    local config_array=("$@")
    for config in "${config_array[@]}"; do
        EXPERIMENTS+=("$config")
    done
}

if [[ -z "$FILTER_MODEL" ]]; then
    # Run all NeMo 25.07 validation configs (matching GB200 Reference exactly)
    add_configs "${LLAMA3_8B_2507_CONFIGS[@]}"
    add_configs "${LLAMA3_70B_2507_CONFIGS[@]}"
    add_configs "${LLAMA31_405B_2507_CONFIGS[@]}"
    add_configs "${DEEPSEEK_V3_LARGE_GBS_2507_CONFIGS[@]}"
    # DEEPSEEK_V3_SMALL_GBS_2507_CONFIGS removed - not in GB200 reference
    add_configs "${QWEN3_30B_2507_CONFIGS[@]}"
    add_configs "${QWEN3_235B_2507_CONFIGS[@]}"
    add_configs "${GPT_OSS_20B_2507_CONFIGS[@]}"
else
    case "$FILTER_MODEL" in
        llama3_8b|llama8b)
            add_configs "${LLAMA3_8B_2507_CONFIGS[@]}"
            ;;
        llama3_70b|llama70b)
            add_configs "${LLAMA3_70B_2507_CONFIGS[@]}"
            ;;
        llama31_405b|llama405b)
            add_configs "${LLAMA31_405B_2507_CONFIGS[@]}"
            ;;
        deepseek_v3|deepseekv3|deepseek|deepseek_large)
            add_configs "${DEEPSEEK_V3_LARGE_GBS_2507_CONFIGS[@]}"
            ;;
        # deepseek_small removed - not in GB200 reference
        qwen3_30b|qwen30b)
            add_configs "${QWEN3_30B_2507_CONFIGS[@]}"
            ;;
        qwen3_30b_tokendrop|qwen30b_tokendrop)
            # Qwen3-30B with token drop enabled
            for config in "${QWEN3_30B_2507_CONFIGS[@]}"; do
                EXPERIMENTS+=("${config} TOKENDROP")
            done
            ;;
        qwen3_30b_tokendrop_mbs4|qwen30b_tokendrop_mbs4)
            # Qwen3-30B with token drop enabled + MBS=4
            for config in "${QWEN3_30B_2507_CONFIGS[@]}"; do
                # Override MBS to 4 (12th field in the config)
                modified_config=$(echo "$config" | awk '{$12=4; print}')
                EXPERIMENTS+=("${modified_config} TOKENDROP")
            done
            ;;
        qwen3_30b_both|qwen30b_both)
            # Qwen3-30B with both dropless and token drop
            add_configs "${QWEN3_30B_2507_CONFIGS[@]}"
            for config in "${QWEN3_30B_2507_CONFIGS[@]}"; do
                EXPERIMENTS+=("${config} TOKENDROP")
            done
            ;;
        qwen3_235b|qwen235b)
            add_configs "${QWEN3_235B_2507_CONFIGS[@]}"
            ;;
        gpt_oss_20b|gptoss20b|gpt_oss|gptoss)
            add_configs "${GPT_OSS_20B_2507_CONFIGS[@]}"
            ;;
        all_qwen)
            add_configs "${QWEN3_30B_2507_CONFIGS[@]}"
            add_configs "${QWEN3_235B_2507_CONFIGS[@]}"
            ;;
        all_llama)
            add_configs "${LLAMA3_8B_2507_CONFIGS[@]}"
            add_configs "${LLAMA3_70B_2507_CONFIGS[@]}"
            add_configs "${LLAMA31_405B_2507_CONFIGS[@]}"
            ;;
        all_gpt_oss)
            add_configs "${GPT_OSS_20B_2507_CONFIGS[@]}"
            ;;
        *)
            echo "Unknown model: $FILTER_MODEL"
            echo "Available models:"
            echo "  llama3_8b, llama3_70b, llama31_405b"
            echo "  deepseek_v3 (GBS=2048 only, matching GB200 reference)"
            echo "  qwen3_30b (dropless), qwen3_30b_tokendrop, qwen3_30b_tokendrop_mbs4, qwen3_30b_both"
            echo "  qwen3_235b"
            echo "  gpt_oss_20b"
            echo "  all_qwen, all_llama, all_gpt_oss"
            exit 1
            ;;
    esac
fi

# ============================================================================
# Run Experiments
# ============================================================================

echo ""
echo "============================================"
echo "NeMo 25.07 Validation Sweep: ${SWEEP_ID}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "============================================"
echo ""
echo "Reference: NeMo 25.07 Official Configurations"
echo "Purpose: Validate Megatron-Bridge performance"
echo ""

# Write header to jobs file
cat > "${JOBS_FILE}" << EOF
# Sweep: ${SWEEP_ID}
# NeMo 25.07 Reference Configuration Validation
# Format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|FSDP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
EOF

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    
    # Parse config - check if TOKENDROP flag is present at the end
    USE_TOKENDROP="False"
    if [[ "$exp" == *" TOKENDROP" ]]; then
        USE_TOKENDROP="True"
        exp="${exp% TOKENDROP}"  # Remove TOKENDROP suffix
    fi
    
    read -r MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG <<< "$exp"
    
    # Calculate derived values
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    DP=$((NUM_GPUS / (TP * PP * CP)))
    
    # Note: For MoE models, EP doesn't need to divide DP.
    # EP distributes experts across GPUs independently.
    # The constraint is: EP * TP * PP * CP <= NUM_GPUS
    # This is automatically satisfied since EP is part of the parallelism config.
    
    # Construct experiment name
    EXP_NAME="nemo2507_${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_tp${TP}pp${PP}cp${CP}ep${EP}dp${DP}"
    if [[ "$TAG" != "nemo2507" ]]; then
        EXP_NAME="${EXP_NAME}_${TAG}"
    fi
    # Add tokendrop suffix if enabled
    if [[ "$USE_TOKENDROP" == "True" ]]; then
        EXP_NAME="${EXP_NAME}_tokendrop"
    fi
    
    echo "[$((i+1))/${#EXPERIMENTS[@]}] Submitting: ${EXP_NAME}"
    echo "  Config: ${NUM_GPUS} GPUs (${NUM_NODES} nodes), SEQ=${SEQ_LEN}"
    echo "  Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, DP=${DP}, VP=${VP}, FSDP=${FSDP}"
    echo "  Batch: MBS=${MBS}, GBS=${GBS}, Precision=${PRECISION}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would submit job"
        echo ""
        continue
    fi
    
    # Get the line count of .slurm_jobs before running
    SLURM_JOBS_FILE="./exp_logs/.slurm_jobs"
    PREV_JOB_COUNT=0
    if [[ -f "${SLURM_JOBS_FILE}" ]]; then
        PREV_JOB_COUNT=$(wc -l < "${SLURM_JOBS_FILE}")
    fi
    
    # Build extra flags for hydra overrides
    EXTRA_FLAGS=""
    
    # Sequence parallel for TP > 1
    if [[ $TP -gt 1 ]]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.sequence_parallel=True"
    fi
    
    # Disable CPU offloading when PP > 1
    if [[ $PP -gt 1 ]]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cpu_offloading_num_layers=0"
        echo "  [INFO] Disabling CPU offloading (incompatible with PP=${PP})"
    fi
    
    # ============================================================================
    # Model-specific overrides to match GB200 Reference exactly
    # ============================================================================
    
    # LLAMA3 70B BF16: Reference requires recompute_layers=20, cpu_offload=0, cuda_graphs=0
    # Base config incorrectly sets cpu_offloading_num_layers=20, we need to fix this
    if [[ "$MODEL_NAME" == "llama3" && "$MODEL_SIZE" == "70b" && "$PRECISION" == "bf16" ]]; then
        echo "  [INFO] Applying LLAMA3 70B BF16 Reference overrides:"
        echo "         - activations_checkpoint_num_layers=20 (recompute)"
        echo "         - cpu_offloading=False, cpu_offloading_num_layers=0"
        echo "         - cuda_graph disabled (FSDP mode)"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.activations_checkpoint_num_layers=20"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cpu_offloading=False"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cpu_offloading_num_layers=0"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cuda_graph_impl=none"
    fi
    
    # LLAMA3 70B FP8: Reference requires FSDP, no recompute, no cpu_offload
    if [[ "$MODEL_NAME" == "llama3" && "$MODEL_SIZE" == "70b" && "$PRECISION" == "fp8" ]]; then
        echo "  [INFO] Applying LLAMA3 70B FP8 Reference overrides:"
        echo "         - No recompute, no cpu_offload"
        echo "         - cuda_graph disabled (FSDP mode)"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.activations_checkpoint_num_layers=0"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cpu_offloading=False"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cpu_offloading_num_layers=0"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cuda_graph_impl=none"
    fi
    
    # LLAMA31 405B: Disable cuda_graphs for PP>1 configs
    if [[ "$MODEL_NAME" == "llama31" && "$MODEL_SIZE" == "405b" && $PP -gt 1 ]]; then
        echo "  [INFO] Disabling cuda_graphs for LLAMA31 405B (PP=${PP})"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cuda_graph_impl=none"
    fi
    
    # Qwen3-30B (MoE): Use transformer_engine CUDA graphs with specific scope
    # Full iteration cuda_graph (local) is incompatible with some MoE operations
    # Use module-level CUDA graphs for: moe_router, moe_preprocess, attn
    # Note: This is specific to Qwen3-30B based on the reference sheet
    if [[ "$MODEL_NAME" == "qwen3" && "$MODEL_SIZE" == "30b_a3b" ]]; then
        echo "  [INFO] Qwen3-30B MoE: Using CUDA graph scope [moe_router,moe_preprocess,attn]"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cuda_graph_impl=transformer_engine"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cuda_graph_scope=\\[moe_router,moe_preprocess,attn\\]"
    fi
    
    # Build VP flag (only if VP > 1)
    VP_FLAG=""
    if [[ $VP -gt 1 ]]; then
        VP_FLAG="-vp ${VP}"
    fi
    
    # Build ETP flag (only if ETP > 1)
    ETP_FLAG=""
    if [[ $ETP -gt 1 ]]; then
        ETP_FLAG="-et ${ETP}"
    fi
    
    # Build FSDP flag (only if FSDP = 1)
    FSDP_FLAG=""
    if [[ $FSDP -eq 1 ]]; then
        FSDP_FLAG="--use_megatron_fsdp true"
    fi
    
    # Build token drop flag
    TOKENDROP_FLAG=""
    if [[ "$USE_TOKENDROP" == "True" ]]; then
        TOKENDROP_FLAG="--use_tokendrop True"
        echo "  [INFO] Token drop ENABLED for this experiment"
    fi
    
    # Run the experiment
    python3 scripts/performance/setup_experiment.py \
        --account ${ACCOUNT} \
        -t ${TIME_LIMIT} \
        --partition ${PARTITION} \
        --gpu ${GPU_TYPE} \
        --model_name ${MODEL_NAME} \
        --model_size ${MODEL_SIZE} \
        --task ${TASK} \
        -ms ${MAX_STEPS} \
        -ng ${NUM_GPUS} \
        -gn ${GPUS_PER_NODE} \
        -c ${PRECISION} \
        -i ${CONTAINER} \
        -hf ${HF_TOKEN} \
        --hf_home ${HF_HOME} \
        --hf_datasets_cache ${HF_DATASETS_CACHE} \
        -wdp ${WANDB_PROJECT} \
        -wdj "${EXP_NAME}" \
        -wdk ${WANDB_KEY} \
        -tp ${TP} \
        -pp ${PP} \
        -cp ${CP} \
        -ep ${EP} \
        ${VP_FLAG} \
        ${ETP_FLAG} \
        ${FSDP_FLAG} \
        ${TOKENDROP_FLAG} \
        -mb ${MBS} \
        -gb ${GBS} \
        ++logger.log_throughput=True \
        ++logger.log_throughput_to_tensorboard=True \
        ++logger.throughput_window_size=5 \
        ++logger.tensorboard_dir="${TENSORBOARD_DIR}" \
        ++logger.tensorboard_log_interval=1 \
        ++logger.log_world_size_to_tensorboard=True \
        ++logger.log_timers_to_tensorboard=True \
        ${EXTRA_FLAGS}
    
    # Extract job ID from .slurm_jobs file
    JOB_ID="unknown"
    EXP_DIR="unknown"
    
    if [[ -f "${SLURM_JOBS_FILE}" ]]; then
        CURR_JOB_COUNT=$(wc -l < "${SLURM_JOBS_FILE}")
        if [[ $CURR_JOB_COUNT -gt $PREV_JOB_COUNT ]]; then
            LAST_LINE=$(tail -1 "${SLURM_JOBS_FILE}")
            JOB_ID=$(echo "${LAST_LINE}" | cut -d' ' -f1)
            EXP_DIR=$(echo "${LAST_LINE}" | grep -oP '"job_dir":\s*"\K[^"]+' || echo "unknown")
        fi
    fi
    
    # Fallback: try to find the most recent matching experiment directory
    if [[ "$EXP_DIR" == "unknown" ]]; then
        EXP_DIR=$(ls -td ./exp_logs/experiments/${MODEL_NAME}_${MODEL_SIZE}_llm_${TASK}_*/  2>/dev/null | head -1 || echo "unknown")
    fi
    
    # Record job info
    echo "${JOB_ID}|${MODEL_NAME}|${MODEL_SIZE}|${NUM_GPUS}|${NUM_NODES}|${GPU_TYPE}|${TP}|${PP}|${CP}|${EP}|${DP}|${VP}|${ETP}|${FSDP}|${SEQ_LEN}|${MBS}|${GBS}|${PRECISION}|${TASK}|${EXP_DIR}" >> "${JOBS_FILE}"
    
    echo "  Submitted: Job ID = ${JOB_ID}"
    echo "  Exp Dir: ${EXP_DIR}"
    echo ""
    
    sleep 2
done

echo "============================================"
echo "All experiments submitted!"
echo "Jobs file: ${JOBS_FILE}"
echo ""
echo "Config file: ${CONFIG_FILE}"
echo ""
echo "To collect and compare results with NeMo 25.07 reference:"
echo ""
echo "  # List available configs"
echo "  python3 collect_results.py --match-config gb200_reference_configs.yaml --list-configs"
echo ""
echo "  # Scan all experiments and match against reference configs"
echo "  python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --use-reference"
echo ""
echo "  # Filter for specific model"
echo "  python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --target-model llama3_8b --use-reference"
echo "  python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --target-model llama3_70b --use-reference"
echo "  python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --target-model deepseek_v3 --use-reference"
echo "  python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --target-model qwen3_30b --use-reference"
echo "  python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --target-model gpt_oss_20b --use-reference"
echo ""
echo "Expected metrics to validate:"
echo "  - Tokens/sec/GPU"
echo "  - TFLOPS per GPU"
echo "  - Iteration time"
echo "  - Memory usage"
echo "============================================"

