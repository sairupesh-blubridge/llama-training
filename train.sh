#!/bin/bash
# Training launcher for Hindi model

# Activate environment
source /workspace/nanotron_env/bin/activate

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Config file path
CONFIG_FILE="/workspace/config_hindi_500m_indic.yaml"
TRAIN_SCRIPT="/workspace/run_train_hindi.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

echo "==========================================="
echo "Starting Hindi Model Training"
echo "==========================================="
echo "GPUs:          $GPU_COUNT"
echo "Config:        $CONFIG_FILE"
echo "Train Script:  $TRAIN_SCRIPT"
echo "Time:          $(date)"
echo "==========================================="

# Create log file with timestamp
LOG_FILE="/workspace/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Run training
cd /workspace
torchrun --nproc_per_node=$GPU_COUNT \
    $TRAIN_SCRIPT \
    --config-file "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"
