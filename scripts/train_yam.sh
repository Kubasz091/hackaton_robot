#!/bin/bash
set -e

# Training script for YAM bimanual policy (ACT)

# Dataset path (assuming mounted at /common/yam_wire or similar if running locally)
# Adjust DATA_DIR to point to where the 'yam_wire' dataset sits.
DATA_DIR=/common/yam_wire

# Output directory
OUTPUT_DIR=outputs/train/act_yam_bimanual

# Use file-based sharing to bypass limited /dev/shm (64MB)
export TORCH_SHARE_STRATEGY=file_system

echo "Starting training..."
python -m lerobot.scripts.lerobot_train \
    --config_path src/lerobot/configs/policy/train_act_yam_bimanual.yaml \
    --dataset.repo_id=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    "$@"

