python3 experimental/send_offline_observation.py \
    --data_dir=observation_data \
    --server_address=localhost:7777 \
    --policy_type=act \
    --pretrained_name_or_path=060000/pretrained_model \
    --task="plug the wire rope into the rubber cable" \
    --policy_device=cuda \
    --actions_per_chunk=100 \
    --viz_joints_address=127.0.0.1:5001