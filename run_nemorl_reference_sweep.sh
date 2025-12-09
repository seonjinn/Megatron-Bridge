#!/bin/bash
#
# NeMo-RL Reference Configuration Training Sweep for Megatron-Bridge
#
# This script runs Megatron-Bridge training with the SAME configurations
# used in NeMo-RL synchronous GRPO training, allowing direct performance comparison.
#
# Source configs: nemo-rl/examples/configs/recipes/llm/performance/grpo-*.yaml (without "off")
#
# Usage: ./run_nemorl_reference_sweep.sh [--dry-run] [--model MODEL_NAME] [--h100|--gb200]
#
# Options:
#   --dry-run       Print commands without executing
#   --model NAME    Run only specific model (deepseek_v3, llama31_8b, qwen3_235b, qwen3_30b, qwen3_32b)
#   --h100          Run on H100 cluster (default)
#   --gb200         Run on GB200 cluster
#

set -e

# Get script directory for finding config files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Parse Arguments
# ============================================================================
DRY_RUN=false
FILTER_MODEL=""
GPU_TYPE="gb200"  # Default to GB200 (4 GPUs per node)

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
        --h100)
            GPU_TYPE="h100"
            shift
            ;;
        --gb200)
            GPU_TYPE="gb200"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--model MODEL_NAME] [--h100|--gb200]"
            echo ""
            echo "Options:"
            echo "  --dry-run       Print commands without executing"
            echo "  --model NAME    Run only specific model:"
            echo "                    deepseek_v3  (256 GPUs, PP=16, EP=16)"
            echo "                    llama31_8b   (16 GPUs, PP=2)"
            echo "                    qwen3_235b   (128 GPUs, TP=2, PP=8, CP=2, EP=16)"
            echo "                    qwen3_30b    (32 GPUs, TP=2, EP=8)"
            echo "                    qwen3_32b    (32 GPUs, TP=4, PP=4)"
            echo "                    all_qwen     (all Qwen models)"
            echo "  --h100          Run on H100 cluster (default, 8 GPUs/node)"
            echo "  --gb200         Run on GB200 cluster (4 GPUs/node)"
            echo ""
            echo "Configs match NeMo-RL synchronous GRPO training settings from:"
            echo "  nemo-rl/examples/configs/recipes/llm/performance/grpo-*.yaml"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Cluster Settings Based on GPU Type
# ============================================================================
if [[ "$GPU_TYPE" == "gb200" ]]; then
    ACCOUNT="coreai_dlalgo_nemorl"
    PARTITION="batch_long"
    GPUS_PER_NODE=4
    CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"
else
    # H100 settings (matching NeMo-RL cluster)
    ACCOUNT="coreai_dlalgo_nemorl"
    PARTITION="batch"
    GPUS_PER_NODE=8
    CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"
fi

TIME_LIMIT="08:00:00"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# HuggingFace settings
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Bridge-NeMoRL-Reference"
TENSORBOARD_DIR="./exp_logs/tensorboard"
MAX_STEPS=100

# Results tracking
SWEEP_ID="nemorl_ref_${GPU_TYPE}_$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="./exp_logs/sweeps/${SWEEP_ID}"
JOBS_FILE="${SWEEP_DIR}/submitted_jobs.txt"

mkdir -p "${SWEEP_DIR}"

# ============================================================================
# NeMo-RL Reference Configurations
# ============================================================================
# Source: nemo-rl/examples/configs/recipes/llm/performance/grpo-*.yaml (without "off")
#
# Format: "MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP MBS GBS PRECISION EXTRA_FLAGS TAG"
#
# Note: We recalculate NUM_NODES based on GPUS_PER_NODE for different clusters

# ============================================================================
# GB200 Configurations from launch_grpo.py presets (4 GPUs/node)
# These match EXACTLY what NeMo-RL runs with `python launch_grpo.py --preset <name>`
# ============================================================================

# LLaMA3.1-8B: 2 nodes x 4 GPUs = 8 GPUs (launch_grpo.py preset: llama8b)
# TP=1, PP=1, EP=1, DP=8, Train GBS=512, SEQ=4096
# Note: PP=1 (not PP=2 from YAML) - override by launch_grpo.py
LLAMA31_8B_CONFIGS=(
    "llama3 8b 8 4096 1 1 1 1 1 0 1 512 bf16 activation_ckpt llama31_8b_nemorl"
)

# LLaMA3.1-70B: 4 nodes x 4 GPUs = 16 GPUs (launch_grpo.py preset: llama70b)
# TP=4, PP=2, EP=1, DP=2, Train GBS=512, SEQ=4096
LLAMA31_70B_CONFIGS=(
    "llama3 70b 16 4096 4 2 1 1 1 0 1 512 bf16 activation_ckpt+seq_parallel llama31_70b_nemorl"
)

# Qwen3-30B-A3B: 4 nodes x 4 GPUs = 16 GPUs (launch_grpo.py preset: qwen30b)
# TP=1, PP=1, EP=8, DP=16, Train GBS=512, SEQ=4096
# Note: TP=1 (not TP=2 from YAML) - override by launch_grpo.py
QWEN3_30B_CONFIGS=(
    "qwen3 30b_a3b 16 4096 1 1 1 8 1 1 1 512 bf16 none qwen3_30b_nemorl"
)

# Qwen3-32B: 4 nodes x 4 GPUs = 16 GPUs (launch_grpo.py preset: qwen32b)
# TP=4, PP=1, EP=1, DP=4, Train GBS=512, SEQ=4096
# Note: PP=1 (not PP=4 from YAML) - override by launch_grpo.py
QWEN3_32B_CONFIGS=(
    "qwen3 32b 16 4096 4 1 1 1 1 0 1 512 bf16 seq_parallel qwen3_32b_nemorl"
)

# ============================================================================
# Models NOT in launch_grpo.py presets - these need custom GB200 configs
# Currently excluded. Add them if needed with appropriate GB200 settings.
# ============================================================================
# DeepSeek-V3: NOT in launch_grpo.py presets (would need custom GB200 config)
DEEPSEEK_V3_CONFIGS=(
    # Keeping original YAML config for reference, but may need adjustment for GB200
    # "deepseek v3 256 1536 1 16 1 16 1 1 1 512 bf16 activation_ckpt deepseek_v3_nemorl"
)

# Qwen3-235B: NOT in launch_grpo.py presets (would need custom GB200 config)
QWEN3_235B_CONFIGS=(
    # Keeping original YAML config for reference, but may need adjustment for GB200
    # "qwen3 235b_a22b 128 8192 2 8 2 16 1 1 1 512 bf16 activation_ckpt+seq_parallel qwen3_235b_nemorl"
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
    # Run all NeMo-RL reference configs (from launch_grpo.py presets)
    add_configs "${LLAMA31_8B_CONFIGS[@]}"
    add_configs "${LLAMA31_70B_CONFIGS[@]}"
    add_configs "${QWEN3_30B_CONFIGS[@]}"
    add_configs "${QWEN3_32B_CONFIGS[@]}"
    # DeepSeek-V3 and Qwen3-235B are NOT in launch_grpo.py presets
else
    case "$FILTER_MODEL" in
        llama31_8b|llama8b|llama3_8b)
            add_configs "${LLAMA31_8B_CONFIGS[@]}"
            ;;
        llama31_70b|llama70b|llama3_70b)
            add_configs "${LLAMA31_70B_CONFIGS[@]}"
            ;;
        qwen3_30b|qwen30b)
            add_configs "${QWEN3_30B_CONFIGS[@]}"
            ;;
        qwen3_32b|qwen32b)
            add_configs "${QWEN3_32B_CONFIGS[@]}"
            ;;
        all_qwen)
            add_configs "${QWEN3_30B_CONFIGS[@]}"
            add_configs "${QWEN3_32B_CONFIGS[@]}"
            ;;
        all_llama)
            add_configs "${LLAMA31_8B_CONFIGS[@]}"
            add_configs "${LLAMA31_70B_CONFIGS[@]}"
            ;;
        *)
            echo "Unknown model: $FILTER_MODEL"
            echo ""
            echo "Available models (from launch_grpo.py GB200 presets):"
            echo "  llama31_8b   - LLaMA3.1-8B (8 GPUs, TP=1, PP=1)"
            echo "  llama31_70b  - LLaMA3.1-70B (16 GPUs, TP=4, PP=2)"
            echo "  qwen3_30b    - Qwen3-30B-A3B (16 GPUs, TP=1, EP=8)"
            echo "  qwen3_32b    - Qwen3-32B (16 GPUs, TP=4, PP=1)"
            echo "  all_qwen     - All Qwen models"
            echo "  all_llama    - All LLaMA models"
            exit 1
            ;;
    esac
fi

# ============================================================================
# Print Summary Table
# ============================================================================

echo ""
echo "============================================"
echo "NeMo-RL Reference Training Sweep: ${SWEEP_ID}"
echo "============================================"
echo ""
echo "GPU Type: ${GPU_TYPE} (${GPUS_PER_NODE} GPUs/node)"
echo "Total experiments: ${#EXPERIMENTS[@]}"
echo ""
echo "Configuration Summary:"
echo "+-----------------+------+-------+----+----+----+----+-------+-------+--------------+"
echo "| Model           | GPUs | Nodes | TP | PP | CP | EP | GBS   | SEQ   | Source       |"
echo "+-----------------+------+-------+----+----+----+----+-------+-------+--------------+"
# GB200 configs from launch_grpo.py presets (4 GPUs/node)
printf "| %-15s | %4d | %5d | %2d | %2d | %2d | %2d | %5d | %5d | %-12s |\n" \
    "LLaMA3.1-8B" 8 2 1 1 1 1 512 4096 "llama8b"
printf "| %-15s | %4d | %5d | %2d | %2d | %2d | %2d | %5d | %5d | %-12s |\n" \
    "LLaMA3.1-70B" 16 4 4 2 1 1 512 4096 "llama70b"
printf "| %-15s | %4d | %5d | %2d | %2d | %2d | %2d | %5d | %5d | %-12s |\n" \
    "Qwen3-30B" 16 4 1 1 1 8 512 4096 "qwen30b"
printf "| %-15s | %4d | %5d | %2d | %2d | %2d | %2d | %5d | %5d | %-12s |\n" \
    "Qwen3-32B" 16 4 4 1 1 1 512 4096 "qwen32b"
echo "+-----------------+------+-------+----+----+----+----+-------+-------+--------------+"
echo ""
echo "⚠️  DeepSeek-V3 and Qwen3-235B are NOT in launch_grpo.py presets"
echo "    Run them separately if needed with custom GB200 configs"
echo "+-----------------+------+-------+----+----+----+----+-------+-------+--------------+"
echo ""
echo "Reference: NeMo-RL synchronous GRPO training configs"
echo "Source: nemo-rl/examples/configs/recipes/llm/performance/grpo-*.yaml"
echo ""

# Write header to jobs file
cat > "${JOBS_FILE}" << EOF
# Sweep: ${SWEEP_ID}
# NeMo-RL Reference Configuration Training (Megatron-Bridge)
# GPU Type: ${GPU_TYPE} (${GPUS_PER_NODE} GPUs/node)
# Format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|SEQ_LEN|MBS|GBS|PRECISION|EXTRA|TAG|EXP_DIR
EOF

# ============================================================================
# Run Experiments
# ============================================================================

for i in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$i]}"
    read -r MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP MBS GBS PRECISION EXTRA_FLAGS TAG <<< "$exp"
    
    # Calculate derived values
    NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
    DP=$((NUM_GPUS / (TP * PP * CP)))
    
    # Construct experiment name
    EXP_NAME="nemorl_${MODEL_NAME}_${MODEL_SIZE}_${PRECISION}_tp${TP}pp${PP}cp${CP}ep${EP}dp${DP}"
    
    echo "[$((i+1))/${#EXPERIMENTS[@]}] Submitting: ${EXP_NAME}"
    echo "  NeMo-RL Source: grpo-${MODEL_NAME}-*.yaml"
    echo "  Config: ${NUM_GPUS} GPUs (${NUM_NODES} ${GPU_TYPE} nodes), SEQ=${SEQ_LEN}"
    echo "  Parallelism: TP=${TP}, PP=${PP}, CP=${CP}, EP=${EP}, DP=${DP}, VP=${VP}"
    echo "  Batch: MBS=${MBS}, GBS=${GBS}, Precision=${PRECISION}"
    echo "  Extra: ${EXTRA_FLAGS}"
    
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
    
    # Build extra Hydra flags based on EXTRA_FLAGS
    HYDRA_FLAGS=""
    
    # ============================================================================
    # NeMo-RL Training Config Overrides - MODEL-SPECIFIC
    # Each model uses different YAML with different optimizer/training settings
    # ============================================================================
    
    # Common settings (all models)
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.optim.name=fused_adam"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.optim.betas=[0.9,0.999]"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.grad_clip=1.0"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.use_distributed_optimizer=True"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.overlap_grad_reduce=True"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.overlap_param_gather=True"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.apply_rope_fusion=True"
    HYDRA_FLAGS="${HYDRA_FLAGS} ++model.bias_activation_fusion=True"
    
    # Model-specific optimizer settings
    if [[ "$MODEL_NAME" == "llama3" ]]; then
        # LLaMA models: grpo-llama3.1-8b-instruct-2n8g.yaml
        echo "  [INFO] Using LLaMA optimizer settings (lr=5e-7, weight_decay=0.0)"
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.optim.lr=5.0e-7"
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.optim.weight_decay=0.0"
        # LLaMA uses activation checkpointing and defer_fp32_logits
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.activations_checkpoint_method=uniform"
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.activations_checkpoint_num_layers=1"
    elif [[ "$MODEL_NAME" == "qwen3" ]]; then
        # Qwen models: grpo-qwen3-*.yaml
        echo "  [INFO] Using Qwen optimizer settings (lr=3e-7, weight_decay=0.01)"
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.optim.lr=3.0e-7"
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.optim.weight_decay=0.01"
        # Qwen30B (MoE) uses PYTORCH_CUDA_ALLOC_CONF and special CUDA graph scope
        if [[ "$MODEL_SIZE" == "30b_a3b" ]]; then
            echo "  [INFO] Qwen30B MoE: Using PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False"
            echo "  [INFO] Qwen30B MoE: Using CUDA graph scope [moe_router,moe_preprocess,attn]"
            # MoE models need transformer_engine cuda graph with specific scope
            HYDRA_FLAGS="${HYDRA_FLAGS} ++model.cuda_graph_impl=transformer_engine"
            HYDRA_FLAGS="${HYDRA_FLAGS} ++model.cuda_graph_scope=\\[moe_router,moe_preprocess,attn\\]"
        fi
    fi
    
    # Sequence parallel for TP > 1
    if [[ $TP -gt 1 ]]; then
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.sequence_parallel=True"
    fi
    
    # Parse EXTRA_FLAGS for additional settings
    if [[ "$EXTRA_FLAGS" == *"activation_ckpt"* ]]; then
        # Already set for LLaMA above, but ensure it's set
        if [[ "$MODEL_NAME" != "llama3" ]]; then
            HYDRA_FLAGS="${HYDRA_FLAGS} ++model.activations_checkpoint_method=uniform"
            HYDRA_FLAGS="${HYDRA_FLAGS} ++model.activations_checkpoint_num_layers=1"
        fi
    fi
    
    if [[ "$EXTRA_FLAGS" == *"seq_parallel"* ]]; then
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.sequence_parallel=True"
    fi
    
    # Disable CPU offloading when PP > 1
    if [[ $PP -gt 1 ]]; then
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.cpu_offloading_num_layers=0"
        HYDRA_FLAGS="${HYDRA_FLAGS} ++model.cuda_graph_impl=none"
        echo "  [INFO] Disabling CPU offloading and cuda_graphs (PP=${PP})"
    fi
    
    # Build VP flag (only if VP > 1)
    VP_FLAG=""
    if [[ $VP -gt 1 ]]; then
        VP_FLAG="-vp ${VP}"
    fi
    
    # Build ETP flag (only if ETP > 0)
    ETP_FLAG=""
    if [[ $ETP -gt 0 ]]; then
        ETP_FLAG="-et ${ETP}"
    fi
    
    # Run the experiment
    python3 scripts/performance/setup_experiment.py \
        --account ${ACCOUNT} \
        -t ${TIME_LIMIT} \
        --partition ${PARTITION} \
        --gpu ${GPU_TYPE} \
        --model_name ${MODEL_NAME} \
        --model_size ${MODEL_SIZE} \
        --task pretrain \
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
        -mb ${MBS} \
        -gb ${GBS} \
        ++data.seq_length=${SEQ_LEN} \
        ++logger.log_throughput=True \
        ++logger.log_throughput_to_tensorboard=True \
        ++logger.throughput_window_size=5 \
        ++logger.tensorboard_dir="${TENSORBOARD_DIR}" \
        ++logger.tensorboard_log_interval=1 \
        ++logger.log_world_size_to_tensorboard=True \
        ++logger.log_timers_to_tensorboard=True \
        ${HYDRA_FLAGS}
    
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
        EXP_DIR=$(ls -td ./exp_logs/experiments/${MODEL_NAME}_${MODEL_SIZE}_llm_pretrain_*/  2>/dev/null | head -1 || echo "unknown")
    fi
    
    # Record job info
    echo "${JOB_ID}|${MODEL_NAME}|${MODEL_SIZE}|${NUM_GPUS}|${NUM_NODES}|${GPU_TYPE}|${TP}|${PP}|${CP}|${EP}|${DP}|${VP}|${ETP}|${SEQ_LEN}|${MBS}|${GBS}|${PRECISION}|${EXTRA_FLAGS}|${TAG}|${EXP_DIR}" >> "${JOBS_FILE}"
    
    echo "  Submitted: Job ID = ${JOB_ID}"
    echo "  Exp Dir: ${EXP_DIR}"
    echo ""
    
    sleep 2
done

echo "============================================"
echo "All experiments submitted!"
echo "Jobs file: ${JOBS_FILE}"
echo ""
echo "Config file: ${SCRIPT_DIR}/nemorl_reference_configs.yaml"
echo ""
echo "To compare results with NeMo-RL reference:"
echo ""
echo "  # List available configs"
echo "  python3 collect_results.py --match-config nemorl_reference_configs.yaml --list-configs"
echo ""
echo "  # Scan all experiments and match against reference configs"
echo "  python3 collect_results.py --scan-all --match-config nemorl_reference_configs.yaml --use-reference"
echo ""
echo "  # Filter for specific model"
echo "  python3 collect_results.py --scan-all --match-config nemorl_reference_configs.yaml --target-model deepseek_v3 --use-reference"
echo "  python3 collect_results.py --scan-all --match-config nemorl_reference_configs.yaml --target-model qwen3_32b --use-reference"
echo ""
echo "Expected metrics to compare:"
echo "  - Tokens/sec/GPU (training throughput)"
echo "  - TFLOPS per GPU"
echo "  - Iteration time"
echo "  - Memory usage"
echo ""
echo "Compare these results with NeMo-RL training metrics!"
echo "============================================"
