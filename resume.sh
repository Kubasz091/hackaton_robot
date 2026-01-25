#!/bin/bash
set -e

mkdir -p /home/hacker/tmp
export TMPDIR=/home/hacker/tmp

# Resume training from the latest checkpoint
# Using 'last' symlink ensures we always get the latest config
.venv/bin/python src/lerobot/scripts/lerobot_train.py \
    --config_path=outputs/train/my_act_3/checkpoints/065000/pretrained_model/train_config.json \
    --policy.type=act \
    --dataset.root=yam_wire_merged \
    --dataset.repo_id=lute/yam_wire_3 \
    --batch_size=64 \
    --steps=100000 \
    --policy.n_action_steps 1\
    --policy.temporal_ensemble_coeff 0.01\
    --output_dir=outputs/train/my_act_4 \
    --job_name=my_smolvla_training \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --num_worker=8 \
    --optimizer.lr=1e-5 \
    --save_freq=3000 \
    --resume=true