#!/bin/bash
#
# GB200 Reference Configuration Sweep Runner
# This script runs experiments with exact configurations from the reference table
# to verify results match expected performance.
#
# Usage: ./run_reference_sweep_gb200.sh [--dry-run] [--model MODEL_NAME] [--precision PRECISION] [--task TASK]
#
# Options:
#   --dry-run           Print commands without executing
#   --model NAME        Run only specific model (llama3_8b, llama3_70b, qwen3_32b, qwen3_30b, qwen3_235b, deepseekv3)
#   --precision PREC    Filter by precision (bf16, fp8_cs, fp8_mx, nvfp4)
#   --task TASK         Filter by task (pretrain, sft, lora)
#

set -e

# ============================================================================
# Parse Arguments
# ============================================================================
DRY_RUN=false
FILTER_MODEL=""
FILTER_PRECISION=""
FILTER_TASK=""

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
        --precision)
            FILTER_PRECISION="$2"
            echo "=== Filtering to precision: ${FILTER_PRECISION} ==="
            shift 2
            ;;
        --task)
            FILTER_TASK="$2"
            echo "=== Filtering to task: ${FILTER_TASK} ==="
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--model MODEL_NAME] [--precision PRECISION] [--task TASK]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Common Settings for GB200
# ============================================================================
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch_long"
TIME_LIMIT="08:00:00"
CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"

# GB200 specific settings
GPU_TYPE="gb200"
GPUS_PER_NODE=4  # GB200 has 4 GPUs per node

# Memory optimization: Help avoid OOM due to fragmentation
# OOM error suggested: expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# HuggingFace settings
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Bridge-Reference"
TENSORBOARD_DIR="./exp_logs/tensorboard"
MAX_STEPS=100

# Results tracking
SWEEP_ID="ref_gb200_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="./exp_logs/sweeps/${SWEEP_ID}"
JOBS_FILE="${SWEEP_DIR}/submitted_jobs.txt"

mkdir -p "${SWEEP_DIR}"

# ============================================================================
# Reference Configurations for GB200 (from official performance table)
# ============================================================================
# Format: "MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK"
#
# GB200 Reference Table Summary:
# ┌─────────────────┬──────┬─────┬────┬────┬────┬────┬────┬─────┬─────┬──────────────────┐
# │ Model           │ GPUs │ TP  │ PP │ CP │ EP │ VP │MBS │ GBS │Step │ Tokens/sec/GPU   │
# ├─────────────────┼──────┼─────┼────┼────┼────┼────┼─────┼─────┼──────────────────┤
# │ DeepSeekV3      │ 256  │  1  │ 4  │ 1  │ 64 │ 4  │ 1  │2048 │10.79│ 3037             │
# │ Qwen3_30B_a3B   │   8  │  1  │ 1  │ 1  │  8 │ 1  │ 4  │ 512 │9.96 │ 26320            │
# │ Qwen3_235B_a22B │  64  │  1  │ 8  │ 1  │  8 │ 1  │ 1  │1024 │14.04│ 4668             │
# │ LLAMA3_8B (SFT) │   8  │  1  │ 1  │ 1  │  - │ -  │ 1  │   8 │0.67 │ 24454            │
# │ LLAMA3_70B (SFT)│  32  │  2  │ 4  │ 1  │  - │ 5  │ 1  │  32 │1.30 │ 3151             │
# │ LLAMA3_70B(LoRA)│   8  │  4  │ 1  │ 1  │  - │20  │ 1  │  64 │5.48 │ 2990             │
# └─────────────────┴──────┴─────┴────┴────┴────┴────┴─────┴─────┴──────────────────┘

declare -A REFERENCE_CONFIGS

# ============================================================================
# DeepSeekV3 (MoE) - GB200 Reference (Pre-train)
# ============================================================================
# BF16:   256 GPUs, TP=1, PP=4, CP=1, DP=64, EP=64, VP=4, ETP=1, FSDP=1, MBS=1, GBS=2048
#         Step: 10.79s, TFLOPs: 793, MFU: 32%, Tokens/sec/GPU: 3037
# FP8-MX: 256 GPUs, same parallelism, Step: 8.97s, TFLOPs: 898, MFU: 18%, Tokens/sec/GPU: 3653
# FP8-MX (drop): Step: 9.94s, TFLOPs: 855, MFU: 17%, Tokens/sec/GPU: 3297
DEEPSEEKV3_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "deepseek v3 256 4096 1 4 1 64 4 1 0 1 2048 bf16 pretrain"
    "deepseek v3 256 4096 1 4 1 64 4 1 0 1 2048 fp8_mx pretrain"
)

# ============================================================================
# Qwen3 32B (Dense) - GB200 Reference (Pre-train)
# ============================================================================
# BF16:   16 GPUs, TP=4, PP=1, CP=1, DP=4, VP=1, ETP=1, FSDP=0, MBS=2, GBS=512
# FP8-CS: 16 GPUs, TP=4, PP=1, CP=1, DP=4, VP=1, ETP=1, FSDP=0, MBS=4, GBS=512
# FP8-MX: 16 GPUs, same as FP8-CS
QWEN3_32B_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "qwen3 32b 16 4096 4 1 1 1 1 1 0 2 512 bf16 pretrain"
    # "qwen3 32b 16 4096 4 1 1 1 1 1 0 4 512 fp8_cs pretrain"
    # "qwen3 32b 16 4096 4 1 1 1 1 1 0 4 512 fp8_mx pretrain"
)

# ============================================================================
# Qwen3 30B A3B (MoE) - GB200 Reference (Pre-train)
# ============================================================================
# BF16:   8 GPUs, TP=1, PP=1, CP=1, DP=8, EP=8, VP=1, ETP=1, FSDP=0, MBS=4, GBS=512
#         Step: 9.96s, TFLOPs: 605, MFU: 25%, Tokens/sec/GPU: 26320
# FP8-MX: 8 GPUs, same parallelism
#         Step: 11.03s, TFLOPs: 547, MFU: 11%, Tokens/sec/GPU: 23766
QWEN3_30B_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "qwen3 30b_a3b 8 4096 1 1 1 8 1 1 0 4 512 bf16 pretrain"
    "qwen3 30b_a3b 8 4096 1 1 1 8 1 1 0 4 512 fp8_mx pretrain"
)

# ============================================================================
# Qwen3 235B A22B (MoE) - GB200 Reference (Pre-train)
# ============================================================================
# BF16:   64 GPUs, TP=1, PP=8, CP=1, DP=8, EP=8, VP=1, ETP=1, FSDP=0, MBS=1, GBS=1024
#         Step: 14.04s, TFLOPs: 691, MFU: 28%, Tokens/sec/GPU: 4668
# FP8-MX: 64 GPUs, same parallelism
#         Step: 15.01s, TFLOPs: 646, MFU: 13%, Tokens/sec/GPU: 4366
# NOTE: Reference uses TP=1, PP=8 (not TP=8!) due to num_query_groups=4 constraint
QWEN3_235B_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "qwen3 235b_a22b 64 4096 1 8 1 8 1 1 0 1 1024 bf16 pretrain"
    "qwen3 235b_a22b 64 4096 1 8 1 8 1 1 0 1 1024 fp8_mx pretrain"
)

# ============================================================================
# LLAMA3 8B - GB200 Reference (Pre-train)
# ============================================================================
# FP8-CS: 8 GPUs, SEQ=8192, TP=1, PP=1, CP=1, DP=8, VP=1, ETP=1, FSDP=0, MBS=2, GBS=128
#         Step: 4.16s, TFLOPs: 1622, MFU: 33%, Tokens/sec/GPU: 29789
# FP8-MX: 8 GPUs, same parallelism
#         Step: 4.4s, TFLOPs: 1533, MFU: 31%
# NVFP4:  8 GPUs, same parallelism
#         Step: 3.25s, TFLOPs: 2076, MFU: 42%, Tokens/sec/GPU: 40330
LLAMA3_8B_PRETRAIN_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "llama3 8b 8 8192 1 1 1 1 1 1 0 2 128 fp8_cs pretrain"
    "llama3 8b 8 8192 1 1 1 1 1 1 0 2 128 fp8_mx pretrain"
    # "llama3 8b 8 8192 1 1 1 1 1 1 0 2 128 nvfp4 pretrain"
)

# ============================================================================
# LLAMA3 8B - GB200 BF16 Reference (SFT)
# ============================================================================
# Reference: 8 GPUs, SEQ=16384, TP=1, PP=1, CP=1, DP=8, VP=1, ETP=1, FSDP=0, MBS=1, GBS=8
# Step time: 0.67s, TFLOPs: 1108, MFU: 45%, Tokens/sec/GPU: 24454
# CG: mlp
LLAMA3_8B_SFT_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "llama3 8b 8 16384 1 1 1 1 1 1 0 1 8 bf16 sft"
)

# ============================================================================
# LLAMA3 70B - GB200 Reference (Pre-train)
# ============================================================================
# FP8-CS: 64 GPUs, SEQ=8192, TP=1, PP=1, CP=1, DP=64, VP=1, ETP=1, FSDP=0, MBS=2, GBS=128
#         recompute_layers=40, cpu_offload=1
#         Step: 3.8s, TFLOPs: 1937, MFU: 40%, Tokens/sec/GPU: 4312
# FP8-MX: 64 GPUs, TP=2, PP=4, CP=1, DP=8, VP=5, ETP=1, FSDP=0, MBS=1, GBS=128
#         Step: 4.53s, TFLOPs: 1625, MFU: 33%, Tokens/sec/GPU: 3617
# NVFP4:  64 GPUs, TP=2, PP=4, CP=1, DP=8, VP=5, ETP=1, FSDP=0, MBS=1, GBS=128
#         Step: 3.95s, TFLOPs: 1863, MFU: 38%, Tokens/sec/GPU: 4148
LLAMA3_70B_PRETRAIN_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "llama3 70b 64 8192 1 1 1 1 1 1 0 2 128 fp8_cs pretrain"
    "llama3 70b 64 8192 2 4 1 1 5 1 0 1 128 fp8_mx pretrain"
    # "llama3 70b 64 8192 2 4 1 1 5 1 0 1 128 nvfp4 pretrain"
)

# ============================================================================
# LLAMA3 70B - GB200 BF16 Reference (SFT)
# ============================================================================
# Reference: 32 GPUs, SEQ=4096, TP=2, PP=4, CP=1, DP=4, VP=5, ETP=1, FSDP=0, MBS=1, GBS=32
# Step time: 1.30s, TFLOPs: 1314, MFU: 54%, Tokens/sec/GPU: 3151
# CG: mlp
LLAMA3_70B_SFT_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "llama3 70b 32 4096 2 4 1 1 5 1 0 1 32 bf16 sft"
)

# ============================================================================
# LLAMA3 70B - GB200 BF16 Reference (LoRA)
# ============================================================================
# Reference: 8 GPUs, SEQ=2048, TP=4, PP=1, CP=1, DP=2, VP=20, ETP=1, FSDP=0, MBS=1, GBS=64
# Step time: 5.48s, TFLOPs: 829, MFU: 34%, Tokens/sec/GPU: 2990
# CG: mlp
LLAMA3_70B_LORA_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
    "llama3 70b 8 2048 4 1 1 1 20 1 0 1 64 bf16 lora"
)

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
    # Run all GB200 reference configs (Pre-train with all dtypes + SFT/LoRA)
    # DeepSeekV3
    add_configs "${DEEPSEEKV3_CONFIGS[@]}"
    # Qwen3
    add_configs "${QWEN3_32B_CONFIGS[@]}"
    add_configs "${QWEN3_30B_CONFIGS[@]}"
    add_configs "${QWEN3_235B_CONFIGS[@]}"
    # LLaMA3 8B
    add_configs "${LLAMA3_8B_PRETRAIN_CONFIGS[@]}"
    add_configs "${LLAMA3_8B_SFT_CONFIGS[@]}"
    # LLaMA3 70B
    add_configs "${LLAMA3_70B_PRETRAIN_CONFIGS[@]}"
    add_configs "${LLAMA3_70B_SFT_CONFIGS[@]}"
    add_configs "${LLAMA3_70B_LORA_CONFIGS[@]}"
else
    case "$FILTER_MODEL" in
        deepseekv3|deepseek)
            add_configs "${DEEPSEEKV3_CONFIGS[@]}"
            ;;
        qwen3_32b|qwen3-32b)
            add_configs "${QWEN3_32B_CONFIGS[@]}"
            ;;
        qwen3_30b|qwen3-30b)
            add_configs "${QWEN3_30B_CONFIGS[@]}"
            ;;
        qwen3_235b|qwen3-235b)
            add_configs "${QWEN3_235B_CONFIGS[@]}"
            ;;
        llama3_8b|llama3-8b)
            # All LLaMA3 8B configs (pretrain + sft)
            add_configs "${LLAMA3_8B_PRETRAIN_CONFIGS[@]}"
            add_configs "${LLAMA3_8B_SFT_CONFIGS[@]}"
            ;;
        llama3_8b_pretrain)
            add_configs "${LLAMA3_8B_PRETRAIN_CONFIGS[@]}"
            ;;
        llama3_8b_sft)
            add_configs "${LLAMA3_8B_SFT_CONFIGS[@]}"
            ;;
        llama3_70b|llama3-70b)
            # All LLaMA3 70B configs (pretrain + sft + lora)
            add_configs "${LLAMA3_70B_PRETRAIN_CONFIGS[@]}"
            add_configs "${LLAMA3_70B_SFT_CONFIGS[@]}"
            add_configs "${LLAMA3_70B_LORA_CONFIGS[@]}"
            ;;
        llama3_70b_pretrain)
            add_configs "${LLAMA3_70B_PRETRAIN_CONFIGS[@]}"
            ;;
        llama3_70b_sft)
            add_configs "${LLAMA3_70B_SFT_CONFIGS[@]}"
            ;;
        llama3_70b_lora)
            add_configs "${LLAMA3_70B_LORA_CONFIGS[@]}"
            ;;
        pretrain)
            # All pretrain configs only
            add_configs "${DEEPSEEKV3_CONFIGS[@]}"
            add_configs "${QWEN3_32B_CONFIGS[@]}"
            add_configs "${QWEN3_30B_CONFIGS[@]}"
            add_configs "${QWEN3_235B_CONFIGS[@]}"
            add_configs "${LLAMA3_8B_PRETRAIN_CONFIGS[@]}"
            add_configs "${LLAMA3_70B_PRETRAIN_CONFIGS[@]}"
            ;;
        *)
            echo "Unknown model: $FILTER_MODEL"
            echo "Available models:"
            echo "  deepseekv3, qwen3_32b, qwen3_30b, qwen3_235b"
            echo "  llama3_8b, llama3_8b_pretrain, llama3_8b_sft"
            echo "  llama3_70b, llama3_70b_pretrain, llama3_70b_sft, llama3_70b_lora"
            echo "  pretrain (all pretrain configs)"
            exit 1
            ;;
    esac
fi

# ============================================================================
# Run Experiments
# ============================================================================

echo ""
echo "============================================"
echo "GB200 Reference Sweep: ${SWEEP_ID}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "============================================"
echo ""

# Write header to jobs file
cat > "${JOBS_FILE}" << EOF
# Sweep: ${SWEEP_ID}
# Reference Configuration Sweep for GB200
# Format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|FSDP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
EOF

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    read -r MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK <<< "$exp"
    
    # Filter by precision if specified
    if [[ -n "$FILTER_PRECISION" ]] && [[ "$PRECISION" != "$FILTER_PRECISION" ]]; then
        continue
    fi
    
    # Filter by task if specified
    if [[ -n "$FILTER_TASK" ]] && [[ "$TASK" != "$FILTER_TASK" ]]; then
        continue
    fi
    
    # Calculate derived values
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    DP=$((NUM_GPUS / (TP * PP * CP)))
    
    # Validate EP divides DP for MoE models
    if [[ $EP -gt 1 ]] && [[ $((DP % EP)) -ne 0 ]]; then
        echo "  [WARNING] EP=$EP does not evenly divide DP=$DP. Skipping."
        continue
    fi
    
    # Construct experiment name
    EXP_NAME="ref_${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_tp${TP}pp${PP}cp${CP}ep${EP}dp${DP}"
    
    echo "[$((i+1))/${#EXPERIMENTS[@]}] Submitting: ${EXP_NAME}"
    echo "  Config: ${NUM_GPUS} GPUs (${NUM_NODES} nodes), SEQ=${SEQ_LEN}"
    echo "  Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, DP=${DP}, VP=${VP}, ETP=${ETP}, FSDP=${FSDP}"
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
    
    # Sequence parallel for TP > 1 (required to avoid validation issues)
    if [[ $TP -gt 1 ]]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.sequence_parallel=True"
    fi
    
    # Disable CPU offloading when PP > 1 (not supported with Pipeline Parallelism)
    if [[ $PP -gt 1 ]]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cpu_offloading_num_layers=0"
        echo "  [INFO] Disabling CPU offloading (incompatible with PP=${PP})"
    fi
    
    # Sequence length override - DISABLED: Use workload_base_config defaults
    # Note: Changing seq_length requires matching model.seq_length and dataset.seq_length
    # If you need custom seq_length, ensure both are set correctly
    # EXTRA_FLAGS="${EXTRA_FLAGS} ++model.seq_length=${SEQ_LEN}"
    # EXTRA_FLAGS="${EXTRA_FLAGS} ++dataset.seq_length=${SEQ_LEN}"
    
    # Precision handling
    case "$PRECISION" in
        fp8_cs)
            EXTRA_FLAGS="${EXTRA_FLAGS} ++train.fp8=True ++train.fp8_recipe=current_scaling"
            ;;
        fp8_mx)
            EXTRA_FLAGS="${EXTRA_FLAGS} ++train.fp8=True ++train.fp8_recipe=mxfp8"
            ;;
        nvfp4)
            EXTRA_FLAGS="${EXTRA_FLAGS} ++train.fp4=True"
            ;;
        bf16)
            # Default, no extra flags needed
            ;;
    esac
    
    # MoE models (EP > 1): Use transformer_engine CUDA graphs with specific scope
    # Full iteration cuda_graph (local) is incompatible with some MoE operations
    # Use module-level CUDA graphs for: moe_router, moe_preprocess, attn
    if [[ $EP -gt 1 ]]; then
        echo "  [INFO] MoE model (EP=${EP}): Using CUDA graph scope [moe_router,moe_preprocess,attn]"
        EXTRA_FLAGS="${EXTRA_FLAGS} ++model.cuda_graph_impl=transformer_engine"
        EXTRA_FLAGS="${EXTRA_FLAGS} '++model.cuda_graph_scope=[moe_router,moe_preprocess,attn]'"
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
    
    # Run the experiment with proper CLI arguments
    # Using CLI args (-tp, -pp, -cp, -ep, -vp, -mb, -gb) instead of hydra overrides
    # for parallelism and batch size settings
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
            # New job was added
            LAST_LINE=$(tail -1 "${SLURM_JOBS_FILE}")
            JOB_ID=$(echo "${LAST_LINE}" | cut -d' ' -f1)
            EXP_DIR=$(echo "${LAST_LINE}" | grep -oP '"job_dir":\s*"\K[^"]+' || echo "unknown")
        else
            echo "  [WARNING] No new job was added to .slurm_jobs file!"
            echo "  [WARNING] Check if the experiment already exists or if there was an error."
        fi
    fi
    
    # Fallback: try to find the most recent matching experiment directory
    if [[ "$EXP_DIR" == "unknown" ]]; then
        EXP_DIR=$(ls -td ./exp_logs/experiments/${MODEL_NAME}_${MODEL_SIZE}_llm_${TASK}_*/  2>/dev/null | head -1 || echo "unknown")
    fi
    
    # Record job info with extended format (including VP, ETP, FSDP, and TASK)
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
echo "To collect results after jobs complete, run:"
echo "  python3 collect_results.py --sweep-dir ${SWEEP_DIR}"
echo ""
echo "To compare with reference values, check:"
echo "  - Tokens/sec/GPU"
echo "  - TFLOPS per GPU"
echo "  - Iteration time (ms)"
echo "============================================"

