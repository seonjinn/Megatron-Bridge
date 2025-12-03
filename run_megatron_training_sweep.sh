#!/bin/bash
#
# RL Training Performance Sweep Runner
# This script measures Megatron-Bridge training performance with RL training configurations.
#
# The goal is to measure training-only performance with parallelism settings
# matching your RL training setup.
#
# Usage: ./run_rl_training_sweep.sh [--dry-run] [--model MODEL_NAME]
#
# Options:
#   --dry-run       Print commands without executing
#   --model NAME    Run only specific model (llama8b, llama70b, llama70b_highseq, qwen30b)
#
# Note: qwen32b is NOT available in Megatron-Bridge. Only qwen30b (Qwen3 30B A3B MoE) is supported.
#

set -e

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
PARTITION="batch"
TIME_LIMIT="00:30:00"
CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"

# GB200 specific settings
GPU_TYPE="gb200"
GPUS_PER_NODE=4  # GB200 has 4 GPUs per node

# HuggingFace settings
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Bridge-RL-Training"
TENSORBOARD_DIR="./exp_logs/tensorboard"
MAX_STEPS=100

# Results tracking
SWEEP_ID="rl_train_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="./exp_logs/sweeps/${SWEEP_ID}"
JOBS_FILE="${SWEEP_DIR}/submitted_jobs.txt"

mkdir -p "${SWEEP_DIR}"

# ============================================================================
# RL Training Configurations
# ============================================================================
# These configurations match your RL training parallelism settings.
#
# Original RL Config Table:
# ┌─────────────────┬──────┬───────┬───────┬───────┬────────┬─────────────────────────┐
# │ Preset          │ GPUs │ Nodes │ R-GBS │ T-GBS │ SeqLen │ Train (TP,CP,EP,PP,DP)  │
# ├─────────────────┼──────┼───────┼───────┼───────┼────────┼─────────────────────────┤
# │ qwen32b         │   16 │     4 │  2048 │   512 │   4096 │ 4,1,1,1,4  (not avail)  │
# │ qwen30b         │   16 │     4 │  2048 │   512 │   4096 │ 1,1,8,1,2               │
# │ llama8b         │    8 │     2 │  2048 │   512 │   4096 │ 1,1,1,1,8               │
# │ llama70b        │   16 │     4 │  2048 │   512 │   4096 │ 4,1,1,2,2               │
# │ llama70b-lowgbs │   16 │     4 │   512 │   512 │   4096 │ 4,1,1,2,2               │
# │ llama70b-highseq│   16 │     4 │  2048 │   512 │  16384 │ 4,1,1,2,2               │
# └─────────────────┴──────┴───────┴───────┴───────┴────────┴─────────────────────────┘
#
# Train parallelism order: (TP, CP, EP, PP, DP)
#
# Note: R-GBS = Rollout GBS (for generation/inference in RL)
#       T-GBS = Training GBS (used in Megatron-Bridge training)
#       For training-only benchmarks, we use T-GBS. R-GBS is shown for reference.
#
# Parallelism verification:
#   - qwen30b: For MoE, world_size = TP × PP × CP × DP × EP = 1×1×1×2×8 = 16 ✓
#   - llama8b: world_size = TP × PP × CP × DP = 1×1×1×8 = 8 ✓
#   - llama70b: world_size = TP × PP × CP × DP = 4×2×1×2 = 16 ✓
#
# Train parallelism order: TP, CP, EP, PP, DP
#
# Note: For MoE models (qwen30b), the table's DP=2 means "data replicas per expert group"
#       Actual DP = world_size / (TP * PP * CP) = 16
#       With EP=8, each expert group has DP/EP = 2 data replicas
#
# Format: "MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG"
# TAG is an optional identifier for special configs (e.g., "lowgbs", "highseq")

# ============================================================================
# LLaMA3 8B - RL Training Config
# ============================================================================
# RL Config: 8 GPUs, 2 Nodes, T-GBS=512, SEQ=4096
# Train parallelism (TP,CP,EP,PP,DP) = (1,1,1,1,8)
# Verification: TP×PP×CP×DP = 1×1×1×8 = 8 GPUs ✓
LLAMA3_8B_RL_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG
    "llama3 8b 8 4096 1 1 1 1 1 1 0 1 512 bf16 pretrain default"
)

# ============================================================================
# LLaMA3 70B - RL Training Config
# ============================================================================
# RL Config: 16 GPUs, 4 Nodes, T-GBS=512, SEQ=4096
# Train parallelism (TP,CP,EP,PP,DP) = (4,1,1,2,2)
# Verification: TP×PP×CP×DP = 4×2×1×2 = 16 GPUs ✓
LLAMA3_70B_RL_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG
    "llama3 70b 16 4096 4 2 1 1 1 1 0 1 512 bf16 pretrain default"
)

# ============================================================================
# LLaMA3 70B Low GBS - RL Training Config
# ============================================================================
# RL Config: 16 GPUs, 4 Nodes, R-GBS=512, T-GBS=512, SEQ=4096
# Train parallelism (TP,CP,EP,PP,DP) = (4,1,1,2,2)
# Note: Same training config as llama70b, only R-GBS differs (for RL rollout)
# Verification: TP×PP×CP×DP = 4×2×1×2 = 16 GPUs ✓
LLAMA3_70B_LOWGBS_RL_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG
    "llama3 70b 16 4096 4 2 1 1 1 1 0 1 512 bf16 pretrain lowgbs"
)

# ============================================================================
# LLaMA3 70B High Sequence - RL Training Config
# ============================================================================
# RL Config: 16 GPUs, 4 Nodes, T-GBS=512, SEQ=16384
# Train parallelism (TP,CP,EP,PP,DP) = (4,1,1,2,2)
# Verification: TP×PP×CP×DP = 4×2×1×2 = 16 GPUs ✓
LLAMA3_70B_HIGHSEQ_RL_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG
    "llama3 70b 16 16384 4 2 1 1 1 1 0 1 512 bf16 pretrain highseq"
)

# ============================================================================
# Qwen3 30B A3B (MoE) - RL Training Config
# ============================================================================
# RL Config: 16 GPUs, 4 Nodes, T-GBS=512, SEQ=4096
# Train parallelism (TP,CP,EP,PP,DP) = (1,1,8,1,2)
# For MoE: world_size = TP×PP×CP×(EP×DP_per_EP) = 1×1×1×(8×2) = 16 GPUs ✓
# Note: In Megatron, we set EP=8 and let DP be auto-calculated as 16
#       DP_per_EP = DP/EP = 16/8 = 2 (matches table's DP=2)
QWEN3_30B_RL_CONFIGS=(
    # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG
    "qwen3 30b_a3b 16 4096 1 1 1 8 1 1 0 1 512 bf16 pretrain default"
)

# ============================================================================
# Qwen 32B - NOT AVAILABLE in Megatron-Bridge
# ============================================================================
# RL Config: 16 GPUs, 4 Nodes, T-GBS=512, SEQ=4096, Train=(TP=4,CP=1,EP=1,PP=2,DP=2)
# Verification: 4 * 1 * 2 * 2 = 16 GPUs ✓
#
# WARNING: Qwen 32B (dense model) is NOT available in Megatron-Bridge.
# Only Qwen3 30B A3B (MoE) and Qwen3 235B A22B (MoE) are supported.
# If you need Qwen 32B, you'll need to add the model config to Megatron-Bridge.
#
# QWEN_32B_RL_CONFIGS=(
#     # MODEL SIZE GPUS SEQ TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK
#     "qwen 32b 16 4096 4 2 1 1 1 1 0 1 512 bf16 pretrain"
# )

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
    # Run all RL training configs
    add_configs "${LLAMA3_8B_RL_CONFIGS[@]}"
    add_configs "${LLAMA3_70B_RL_CONFIGS[@]}"
    add_configs "${LLAMA3_70B_LOWGBS_RL_CONFIGS[@]}"
    add_configs "${LLAMA3_70B_HIGHSEQ_RL_CONFIGS[@]}"
    add_configs "${QWEN3_30B_RL_CONFIGS[@]}"
else
    case "$FILTER_MODEL" in
        llama8b|llama3_8b)
            add_configs "${LLAMA3_8B_RL_CONFIGS[@]}"
            ;;
        llama70b|llama3_70b)
            add_configs "${LLAMA3_70B_RL_CONFIGS[@]}"
            ;;
        llama70b_lowgbs|llama3_70b_lowgbs|llama70b-lowgbs)
            add_configs "${LLAMA3_70B_LOWGBS_RL_CONFIGS[@]}"
            ;;
        llama70b_highseq|llama3_70b_highseq|llama70b-highseq)
            add_configs "${LLAMA3_70B_HIGHSEQ_RL_CONFIGS[@]}"
            ;;
        qwen30b|qwen3_30b)
            add_configs "${QWEN3_30B_RL_CONFIGS[@]}"
            ;;
        all_llama70b)
            # All LLaMA 70B variants (normal, lowgbs, highseq)
            add_configs "${LLAMA3_70B_RL_CONFIGS[@]}"
            add_configs "${LLAMA3_70B_LOWGBS_RL_CONFIGS[@]}"
            add_configs "${LLAMA3_70B_HIGHSEQ_RL_CONFIGS[@]}"
            ;;
        *)
            echo "Unknown model: $FILTER_MODEL"
            echo "Available models:"
            echo "  llama8b (LLaMA3 8B)"
            echo "  llama70b (LLaMA3 70B)"
            echo "  llama70b_lowgbs (LLaMA3 70B with low R-GBS=512)"
            echo "  llama70b_highseq (LLaMA3 70B with SEQ=16384)"
            echo "  qwen30b (Qwen3 30B A3B MoE)"
            echo "  all_llama70b (all llama70b variants)"
            echo ""
            echo "Note: qwen32b is NOT available in Megatron-Bridge"
            exit 1
            ;;
    esac
fi

# ============================================================================
# Run Experiments
# ============================================================================

echo ""
echo "============================================"
echo "RL Training Performance Sweep: ${SWEEP_ID}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "============================================"
echo ""
echo "RL Training Config Summary (BF16 only):"
echo "┌───────────────────┬──────┬───────┬───────┬───────────────────────────┬───────┐"
echo "│ Model             │ GPUs │ R-GBS │ T-GBS │ Train (TP,CP,EP,PP,DP)    │ SEQ   │"
echo "├───────────────────┼──────┼───────┼───────┼───────────────────────────┼───────┤"
echo "│ LLaMA3 8B         │    8 │  2048 │   512 │ 1,1,1,1,8                 │  4096 │"
echo "│ LLaMA3 70B        │   16 │  2048 │   512 │ 4,1,1,2,2                 │  4096 │"
echo "│ LLaMA3 70B lowgbs │   16 │   512 │   512 │ 4,1,1,2,2                 │  4096 │"
echo "│ LLaMA3 70B HS     │   16 │  2048 │   512 │ 4,1,1,2,2                 │ 16384 │"
echo "│ Qwen3 30B (MoE)   │   16 │  2048 │   512 │ 1,1,8,1,2 (DP_per_EP=2)   │  4096 │"
echo "└───────────────────┴──────┴───────┴───────┴───────────────────────────┴───────┘"
echo ""

# Write header to jobs file
cat > "${JOBS_FILE}" << EOF
# Sweep: ${SWEEP_ID}
# RL Training Performance Sweep
# Format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|FSDP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
EOF

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    read -r MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG <<< "$exp"
    
    # Calculate derived values
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    DP=$((NUM_GPUS / (TP * PP * CP)))
    
    # Validate EP divides DP for MoE models
    if [[ $EP -gt 1 ]] && [[ $((DP % EP)) -ne 0 ]]; then
        echo "  [WARNING] EP=$EP does not evenly divide DP=$DP. Skipping."
        continue
    fi
    
    # Construct experiment name with unique identifier
    EXP_NAME="rl_${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_tp${TP}pp${PP}cp${CP}ep${EP}dp${DP}_gbs${GBS}_seq${SEQ_LEN}"
    
    # Add tag suffix for special configs (e.g., lowgbs, highseq)
    if [[ "$TAG" != "default" ]]; then
        EXP_NAME="${EXP_NAME}_${TAG}"
    fi
    
    echo "[$((i+1))/${#EXPERIMENTS[@]}] Submitting: ${EXP_NAME}"
    echo "  Config: ${NUM_GPUS} GPUs (${NUM_NODES} nodes), SEQ=${SEQ_LEN}, GBS=${GBS}"
    echo "  Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, DP=${DP}, VP=${VP}"
    echo "  Precision: ${PRECISION}"
    
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
    
    # Sequence length override
    EXTRA_FLAGS="${EXTRA_FLAGS} ++data.seq_length=${SEQ_LEN}"
    
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
        LAST_LINE=$(tail -1 "${SLURM_JOBS_FILE}")
        JOB_ID=$(echo "${LAST_LINE}" | cut -d' ' -f1)
        EXP_DIR=$(echo "${LAST_LINE}" | grep -oP '"job_dir":\s*"\K[^"]+' || echo "unknown")
    fi
    
    if [[ "$EXP_DIR" == "unknown" ]]; then
        EXP_DIR=$(ls -td ./exp_logs/experiments/${MODEL_NAME}_${MODEL_SIZE}_*/  2>/dev/null | head -1 || echo "unknown")
    fi
    
    # Record job info with extended format
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
echo "Expected metrics to compare with RL training:"
echo "  - Tokens/sec/GPU (training throughput)"
echo "  - TFLOPS per GPU"
echo "  - Iteration time (Step time)"
echo "============================================"

