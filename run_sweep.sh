#!/bin/bash
#
# Multi-experiment sweep runner
# Usage: ./run_sweep.sh [--dry-run]
#
# This script runs multiple experiments with different models and parallelism configurations.
# Results are collected after all jobs complete.
#

set -e

# ============================================================================
# Configuration
# ============================================================================

# Dry run mode (just print commands, don't execute)
DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ==="
fi

# Common settings
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
TIME_LIMIT="00:30:00"
CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"

# GPU configuration
# GPU_TYPE: gb200, gb300, h100, b200, etc.
# GPUS_PER_NODE: gb200=4, gb300=4, h100=8, b200=8
GPU_TYPE="gb200"
GPUS_PER_NODE=4  # GB200: 4 GPUs/node, H100: 8 GPUs/node

# HuggingFace settings (use environment variables if set, otherwise use defaults)
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Standalone"
PRECISION="bf16"
MAX_STEPS=100

# Batch size settings (optional - leave empty to use model defaults)
# Set these to override default batch sizes (useful for OOM issues)
GLOBAL_BATCH_SIZE="512"      # e.g., 512, 1024, 2048
MICRO_BATCH_SIZE=""       # e.g., 1, 2, 4

# Results tracking
SWEEP_ID=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR="./exp_logs/sweeps/${SWEEP_ID}"
JOBS_FILE="${SWEEP_DIR}/submitted_jobs.txt"

mkdir -p "${SWEEP_DIR}"
echo "Sweep ID: ${SWEEP_ID}"
echo "Results will be saved to: ${SWEEP_DIR}"

# ============================================================================
# Define Experiments
# ============================================================================
# Format: "MODEL_NAME MODEL_SIZE NUM_GPUS TP PP CP EP"
# DP is calculated automatically: DP = NUM_GPUS / (TP * PP * CP)
#
# Add your experiment configurations here:

EXPERIMENTS=(
    # ============================================
    # Llama3 8B experiments
    # ============================================
    "llama3 8b 16 1 1 1 1"   # DP=16 (pure data parallel)
    "llama3 8b 16 2 1 1 1"   # TP=2, DP=8
    "llama3 8b 16 4 1 1 1"   # TP=4, DP=4
    
    # ============================================
    # Qwen3 30B MoE (Active 3B) experiments
    # Recommended: EP=8 for 8 experts
    # NOTE: TP>1 requires sequence_parallel=True (auto-enabled below)
    # ============================================
    "qwen3 30b_a3b 8 1 1 1 8"    # 8 GPUs, TP=1, EP=8, DP=1
    "qwen3 30b_a3b 16 1 1 1 8"   # 16 GPUs, TP=1, EP=8, DP=2
    "qwen3 30b_a3b 16 2 1 1 8"   # 16 GPUs, TP=2, EP=8, DP=1 (SP=True)
    "qwen3 30b_a3b 32 1 1 1 8"   # 32 GPUs, TP=1, EP=8, DP=4
    "qwen3 30b_a3b 32 2 1 1 8"   # 32 GPUs, TP=2, EP=8, DP=2 (SP=True)
    
    # ============================================
    # Qwen3 235B MoE (Active 22B) experiments
    # Recommended: EP=64 for 64 experts, needs more GPUs
    # ============================================
    # "qwen3 235b_a22b 64 2 1 1 64"   # 64 GPUs, TP=2, EP=64 (GB300 config)
    # "qwen3 235b_a22b 64 1 8 1 8"    # 64 GPUs, PP=8, EP=8 (GB200 config)
    # "qwen3 235b_a22b 128 2 1 1 64"  # 128 GPUs, TP=2, EP=64, DP=2
    
    # ============================================
    # Llama3 70B experiments (needs more parallelism)
    # ============================================
    # "llama3 70b 32 4 2 1 1"  # 32 GPUs, TP=4, PP=2, DP=4
    # "llama3 70b 64 8 2 1 1"  # 64 GPUs, TP=8, PP=2, DP=4
)

# ============================================================================
# Run Experiments
# ============================================================================

echo ""
echo "============================================"
echo "Starting Sweep: ${SWEEP_ID}"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo "============================================"
echo ""

# Write header to jobs file
echo "# Sweep: ${SWEEP_ID}" > "${JOBS_FILE}"
echo "# Format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|EXP_DIR" >> "${JOBS_FILE}"
echo "" >> "${JOBS_FILE}"

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    read -r MODEL_NAME MODEL_SIZE NUM_GPUS TP PP CP EP <<< "$exp"
    
    # Calculate derived values
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    
    # DP calculation:
    # Total GPUs = TP * PP * CP * DP
    # For MoE models, EP is a subdivision of DP (EP <= DP)
    # DP = NUM_GPUS / (TP * PP * CP)
    # DP_per_expert_group = DP / EP (number of data parallel replicas per expert group)
    DP=$((NUM_GPUS / (TP * PP * CP)))
    
    # Validate: EP should divide DP evenly for MoE models
    if [[ $EP -gt 1 ]] && [[ $((DP % EP)) -ne 0 ]]; then
        echo "  [WARNING] EP=$EP does not evenly divide DP=$DP. This may cause issues."
    fi
    
    # Construct experiment name
    EXP_NAME="${MODEL_NAME}-${MODEL_SIZE}-${NUM_GPUS}gpus-tp${TP}-pp${PP}-cp${CP}-ep${EP}-dp${DP}"
    
    echo "[$((i+1))/${#EXPERIMENTS[@]}] Submitting: ${EXP_NAME}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [DRY RUN] Would run:"
        echo "    MODEL=${MODEL_NAME} SIZE=${MODEL_SIZE} GPUS=${NUM_GPUS}"
        echo "    TP=${TP} PP=${PP} CP=${CP} EP=${EP} DP=${DP}"
        echo ""
        continue
    fi
    
    # Get the line count of .slurm_jobs before running to find new job
    SLURM_JOBS_FILE="./exp_logs/.slurm_jobs"
    PREV_JOB_COUNT=0
    if [[ -f "${SLURM_JOBS_FILE}" ]]; then
        PREV_JOB_COUNT=$(wc -l < "${SLURM_JOBS_FILE}")
    fi
    
    # Determine if sequence parallel should be enabled (required when TP > 1)
    if [[ $TP -gt 1 ]]; then
        SP_FLAG="++model.sequence_parallel=True"
    else
        SP_FLAG=""
    fi
    
    # Build batch size flags (only if set)
    BATCH_FLAGS=""
    if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
        BATCH_FLAGS="${BATCH_FLAGS} ++train.global_batch_size=${GLOBAL_BATCH_SIZE}"
    fi
    if [[ -n "${MICRO_BATCH_SIZE}" ]]; then
        BATCH_FLAGS="${BATCH_FLAGS} ++train.micro_batch_size=${MICRO_BATCH_SIZE}"
    fi
    
    # Run the experiment
    python3 scripts/performance/setup_experiment.py \
        --account ${ACCOUNT} \
        -t ${TIME_LIMIT} \
        --partition ${PARTITION} \
        --gpu ${GPU_TYPE} \
        --model_name ${MODEL_NAME} \
        --model_size ${MODEL_SIZE} \
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
        ++logger.log_throughput=True \
        ++logger.log_throughput_to_tensorboard=True \
        ++logger.throughput_window_size=5 \
        ++logger.log_world_size_to_tensorboard=True \
        ++logger.log_timers_to_tensorboard=True \
        ${SP_FLAG} \
        ${BATCH_FLAGS}
    
    # Extract job ID from .slurm_jobs file (nemo_run appends new jobs here)
    # Format: JOB_ID = /path/to/experiment/...
    JOB_ID="unknown"
    EXP_DIR="unknown"
    
    if [[ -f "${SLURM_JOBS_FILE}" ]]; then
        # Get the last line (newest job)
        LAST_LINE=$(tail -1 "${SLURM_JOBS_FILE}")
        # Extract job ID (first field before " = ")
        JOB_ID=$(echo "${LAST_LINE}" | cut -d' ' -f1)
        # Extract experiment directory from job_dir in JSON
        EXP_DIR=$(echo "${LAST_LINE}" | grep -oP '"job_dir":\s*"\K[^"]+' || echo "unknown")
    fi
    
    # Fallback: find latest experiment directory if extraction failed
    if [[ "$EXP_DIR" == "unknown" ]]; then
        EXP_DIR=$(ls -td ./exp_logs/experiments/${MODEL_NAME}_${MODEL_SIZE}_*/  2>/dev/null | head -1 || echo "unknown")
    fi
    
    # Record job info
    echo "${JOB_ID}|${MODEL_NAME}|${MODEL_SIZE}|${NUM_GPUS}|${NUM_NODES}|${GPU_TYPE}|${TP}|${PP}|${CP}|${EP}|${DP}|${EXP_DIR}" >> "${JOBS_FILE}"
    
    echo "  Submitted: Job ID = ${JOB_ID}"
    echo "  Exp Dir: ${EXP_DIR}"
    echo ""
    
    # Small delay between submissions
    sleep 2
done

echo "============================================"
echo "All experiments submitted!"
echo "Jobs file: ${JOBS_FILE}"
echo ""
echo "To collect results after jobs complete, run:"
echo "  python3 collect_results.py --sweep-dir ${SWEEP_DIR}"
echo "============================================"

