#!/bin/bash
#
# Batch Experiment Runner for Megatron-Bridge
# Runs multiple experiments with different models and parallelism configurations
#

set -e

# ============================================================================
# Configuration
# ============================================================================

# Common settings
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
TIME_LIMIT="00:30:00"
CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"
WANDB_KEY="cd4db01aafd020369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Standalone"
PRECISION="bf16"
MAX_STEPS=100
GPUS_PER_NODE=4

# Output directory for experiment tracking
BATCH_RESULTS_DIR="./exp_logs/batch_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_ID="batch_${TIMESTAMP}"
BATCH_LOG_FILE="${BATCH_RESULTS_DIR}/${BATCH_ID}/experiments.log"

# ============================================================================
# Experiment Configurations
# Define your experiments here as: "MODEL_NAME:MODEL_SIZE:NUM_GPUS:TP:PP:CP:EP"
# ============================================================================

EXPERIMENTS=(
    # Llama3 8B experiments with different parallelism
    "llama3:8b:8:1:1:1:1"      # 8 GPUs, no parallelism (DP only)
    "llama3:8b:8:2:1:1:1"      # 8 GPUs, TP=2
    "llama3:8b:8:4:1:1:1"      # 8 GPUs, TP=4
    "llama3:8b:16:1:1:1:1"     # 16 GPUs, DP only
    "llama3:8b:16:2:1:1:1"     # 16 GPUs, TP=2
    "llama3:8b:16:4:1:1:1"     # 16 GPUs, TP=4
    "llama3:8b:16:2:2:1:1"     # 16 GPUs, TP=2, PP=2
    "llama3:8b:16:4:2:1:1"     # 16 GPUs, TP=4, PP=2
    
    # Add more experiments as needed:
    # "llama3:70b:64:8:2:1:1"
    # "mistral:7b:8:1:1:1:1"
)

# ============================================================================
# Functions
# ============================================================================

log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}" | tee -a "${BATCH_LOG_FILE}"
}

run_single_experiment() {
    local config="$1"
    
    # Parse configuration
    IFS=':' read -r MODEL_NAME MODEL_SIZE NUM_GPUS TP PP CP EP <<< "${config}"
    
    # Calculate derived values
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    DP=$((NUM_GPUS / (TP * PP * CP)))
    
    # Construct experiment name
    EXP_NAME="${MODEL_NAME}-${MODEL_SIZE}-${NUM_GPUS}gpus-${NUM_NODES}nodes-tp${TP}-pp${PP}-cp${CP}-ep${EP}-dp${DP}"
    
    log_message "Starting experiment: ${EXP_NAME}"
    log_message "  Model: ${MODEL_NAME} ${MODEL_SIZE}"
    log_message "  GPUs: ${NUM_GPUS} (${NUM_NODES} nodes × ${GPUS_PER_NODE} GPUs/node)"
    log_message "  Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, DP=${DP}"
    
    # Run the experiment
    python3 scripts/performance/setup_experiment.py \
        --account ${ACCOUNT} \
        -t ${TIME_LIMIT} \
        --partition ${PARTITION} \
        --gpu gb200 \
        --model_name ${MODEL_NAME} \
        --model_size ${MODEL_SIZE} \
        -ms ${MAX_STEPS} \
        -ng ${NUM_GPUS} \
        -gn ${GPUS_PER_NODE} \
        -c ${PRECISION} \
        -i ${CONTAINER} \
        -hf ${HF_TOKEN} \
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
        2>&1 | tee -a "${BATCH_LOG_FILE}"
    
    # Record experiment info
    echo "${EXP_NAME}:${config}:$(date +%s)" >> "${BATCH_RESULTS_DIR}/${BATCH_ID}/submitted_jobs.txt"
    
    log_message "Submitted experiment: ${EXP_NAME}"
    log_message "---"
    
    # Optional: Add delay between submissions to avoid overwhelming the scheduler
    sleep 2
}

# ============================================================================
# Main
# ============================================================================

main() {
    # Create output directories
    mkdir -p "${BATCH_RESULTS_DIR}/${BATCH_ID}"
    
    log_message "=========================================="
    log_message "Starting Batch Experiments: ${BATCH_ID}"
    log_message "Total experiments: ${#EXPERIMENTS[@]}"
    log_message "=========================================="
    
    # Save experiment configurations
    printf '%s\n' "${EXPERIMENTS[@]}" > "${BATCH_RESULTS_DIR}/${BATCH_ID}/experiment_configs.txt"
    
    # Run each experiment
    local exp_count=0
    for config in "${EXPERIMENTS[@]}"; do
        exp_count=$((exp_count + 1))
        log_message "Experiment ${exp_count}/${#EXPERIMENTS[@]}"
        run_single_experiment "${config}"
    done
    
    log_message "=========================================="
    log_message "All experiments submitted!"
    log_message "Results directory: ${BATCH_RESULTS_DIR}/${BATCH_ID}"
    log_message "=========================================="
    log_message ""
    log_message "To collect results after experiments complete, run:"
    log_message "  python3 scripts/performance/collect_results.py --batch-id ${BATCH_ID}"
}

# Run main function
main "$@"

