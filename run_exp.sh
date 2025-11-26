#!/bin/bash

# Basic experiment parameters
MODEL_NAME="llama3"
MODEL_SIZE="8b"
NUM_GPUS=16
GPUS_PER_NODE=4
PRECISION="bf16"
TIME_LIMIT="00:20:00"
ACCOUNT="coreai_dlalgo_nemorl"
PARTITION="batch"
CONTAINER="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/Megatron-Bridge/nemo_25.11.rc6.sqsh"

# HuggingFace settings (use environment variables if set, otherwise use defaults)
HF_TOKEN="${HF_TOKEN:-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}"
HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/cache}"

# WandB settings
WANDB_KEY="cd4db01aafd025d20369f8eee65e6292c28bfe0d"
WANDB_PROJECT="Megatron-Standalone"
TENSORBOARD_DIR="./exp_logs/tensorboard"

# Parallelism settings (Defaults to 1 if not set)
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-1}

# Calculate derived values
NUM_NODES=$((NUM_GPUS / GPUS_PER_NODE))
DP=$((NUM_GPUS / (TP * PP * CP)))


MAX_STEPS=${MAX_STEPS:-100} # Default to 100 if not set


# Construct WandB Experiment Name
WANDB_EXP_NAME="${MODEL_NAME}-${MODEL_SIZE}-${NUM_GPUS}gpus-${NUM_NODES}nodes-tp${TP}-pp${PP}-cp${CP}-ep${EP}-dp${DP}"

echo "Running experiment: ${WANDB_EXP_NAME}"

# Run setup_experiment.py
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
  --hf_home ${HF_HOME} \
  --hf_datasets_cache ${HF_DATASETS_CACHE} \
  -wdp ${WANDB_PROJECT} \
  -wdj "${WANDB_EXP_NAME}" \
  -wdk ${WANDB_KEY} \
  -tp ${TP} \
  -pp ${PP} \
  -cp ${CP} \
  -ep ${EP} \
  ++logger.log_throughput=True \
  ++logger.log_throughput_to_tensorboard=True \
  ++logger.throughput_window_size=5 \
  ++logger.tensorboard_dir="${TENSORBOARD_DIR}" \
  ++logger.tensorboard_log_interval=1 \
  ++logger.log_world_size_to_tensorboard=True \
  ++logger.log_timers_to_tensorboard=True 