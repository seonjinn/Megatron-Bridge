#!/bin/bash
#
# Qwen3 30B A3B Token Drop Comparison Experiment
# This script runs two experiments:
#   1. Dropless mode (default, moe_token_dropping=false)
#   2. Token drop mode (moe_token_dropping=true with capacity factor)
#
# Usage: ./run_qwen30b_tokendrop_comparison.sh [--dry-run]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Parse Arguments
# ============================================================================
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            echo "=== DRY RUN MODE ==="
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run]"
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
GPUS_PER_NODE=4

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# HuggingFace settings
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"

WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Bridge-TokenDrop-Comparison"
TENSORBOARD_DIR="./exp_logs/tensorboard"
MAX_STEPS=50

# Results tracking
SWEEP_ID="qwen30b_tokendrop_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="./exp_logs/sweeps/${SWEEP_ID}"
JOBS_FILE="${SWEEP_DIR}/submitted_jobs.txt"

mkdir -p "${SWEEP_DIR}"

# ============================================================================
# Qwen3 30B A3B Configuration (from NeMo 25.07 reference)
# ============================================================================
# pre_train,qwen3,30b_a3b,gb200,bf16,8,4096,1,1,1,8,1,1,512,1,1,0,0,0
MODEL_NAME="qwen3"
MODEL_SIZE="30b_a3b"
NUM_GPUS=8
SEQ_LEN=4096
TP=1
PP=1
CP=1
EP=8
VP=1
ETP=1
FSDP=0
MBS=1
GBS=512
PRECISION="bf16"
TASK="pretrain"

NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))

# ============================================================================
# Common EXTRA_FLAGS for Qwen3 30B
# ============================================================================
# CUDA Graph settings specific to Qwen3-30B (moe_router, moe_preprocess, attn)
COMMON_EXTRA_FLAGS="++model.cuda_graph_impl=transformer_engine ++model.cuda_graph_scope=\\[moe_router,moe_preprocess,attn\\]"
COMMON_EXTRA_FLAGS="${COMMON_EXTRA_FLAGS} ++env_vars.PYTORCH_CUDA_ALLOC_CONF=\"expandable_segments:False\""

echo "============================================================================"
echo "Qwen3 30B A3B Token Drop Comparison"
echo "============================================================================"
echo "Configuration:"
echo "  Model: ${MODEL_NAME}_${MODEL_SIZE}"
echo "  GPUs: ${NUM_GPUS} (${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs)"
echo "  Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}"
echo "  Batch: MBS=${MBS}, GBS=${GBS}"
echo "  Precision: ${PRECISION}"
echo "  Max Steps: ${MAX_STEPS}"
echo ""
echo "Experiments:"
echo "  1. Dropless mode (use_tokendrop=False)"
echo "  2. Token drop mode (use_tokendrop=True)"
echo "============================================================================"

# ============================================================================
# Experiment 1: Dropless Mode (default)
# ============================================================================
echo ""
echo ">>> Submitting Experiment 1: Dropless Mode"
WANDB_EXP_NAME_1="qwen30b_dropless_tp${TP}pp${PP}cp${CP}ep${EP}_gbs${GBS}"

CMD_DROPLESS="python3 scripts/performance/setup_experiment.py \
    --account ${ACCOUNT} \
    --partition ${PARTITION} \
    --time_limit ${TIME_LIMIT} \
    --container_image ${CONTAINER} \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --gpu ${GPU_TYPE} \
    --num_gpus ${NUM_GPUS} \
    --gpus_per_node ${GPUS_PER_NODE} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --expert_model_parallel_size ${EP} \
    --micro_batch_size ${MBS} \
    --global_batch_size ${GBS} \
    --seq_length ${SEQ_LEN} \
    --max_steps ${MAX_STEPS} \
    --compute_dtype ${PRECISION} \
    --task ${TASK} \
    --hf_token ${HF_TOKEN} \
    --wandb_key ${WANDB_KEY} \
    --wandb_prj_name ${WANDB_PROJECT} \
    --wandb_exp_name ${WANDB_EXP_NAME_1} \
    --log_dir ./exp_logs/experiments \
    --use_tokendrop False \
    ${COMMON_EXTRA_FLAGS}"

echo "Command: ${CMD_DROPLESS}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] Would execute above command"
else
    eval "${CMD_DROPLESS}"
    echo "  Submitted: ${WANDB_EXP_NAME_1}" | tee -a "${JOBS_FILE}"
fi

# ============================================================================
# Experiment 2: Token Drop Mode
# ============================================================================
echo ""
echo ">>> Submitting Experiment 2: Token Drop Mode"
WANDB_EXP_NAME_2="qwen30b_tokendrop_tp${TP}pp${PP}cp${CP}ep${EP}_gbs${GBS}"

CMD_TOKENDROP="python3 scripts/performance/setup_experiment.py \
    --account ${ACCOUNT} \
    --partition ${PARTITION} \
    --time_limit ${TIME_LIMIT} \
    --container_image ${CONTAINER} \
    --model_name ${MODEL_NAME} \
    --model_size ${MODEL_SIZE} \
    --gpu ${GPU_TYPE} \
    --num_gpus ${NUM_GPUS} \
    --gpus_per_node ${GPUS_PER_NODE} \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --context_parallel_size ${CP} \
    --expert_model_parallel_size ${EP} \
    --micro_batch_size ${MBS} \
    --global_batch_size ${GBS} \
    --seq_length ${SEQ_LEN} \
    --max_steps ${MAX_STEPS} \
    --compute_dtype ${PRECISION} \
    --task ${TASK} \
    --hf_token ${HF_TOKEN} \
    --wandb_key ${WANDB_KEY} \
    --wandb_prj_name ${WANDB_PROJECT} \
    --wandb_exp_name ${WANDB_EXP_NAME_2} \
    --log_dir ./exp_logs/experiments \
    --use_tokendrop True \
    ${COMMON_EXTRA_FLAGS}"

echo "Command: ${CMD_TOKENDROP}"
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY RUN] Would execute above command"
else
    eval "${CMD_TOKENDROP}"
    echo "  Submitted: ${WANDB_EXP_NAME_2}" | tee -a "${JOBS_FILE}"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "Sweep Complete: ${SWEEP_ID}"
echo "============================================================================"
echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Results will be logged to WandB project: ${WANDB_PROJECT}"
echo ""
echo "To compare results after completion:"
echo "  python3 collect_results.py --scan-all --target-model qwen3_30b"
echo ""

