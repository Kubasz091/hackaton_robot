#!/usr/bin/env python

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD

def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy

def visualize_dataset(
    dataset: LeRobotDataset,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    repo_id: str = "dataset"
) -> Path | None:
    
    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False 
    )

    logging.info("Starting Rerun")
    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}_multi_ep", spawn=spawn_local_viewer)
    gc.collect()

    if mode == "distant":
        rr.serve_web_viewer(open_browser=False, web_port=web_port)

    logging.info("Logging to Rerun")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        for i in range(len(batch["index"])):
            # Get episode index to separate timelines/entities
            # Note: LeRobotDataset returns 'episode_index' as a tensor in the batch
            ep_idx = batch["episode_index"][i].item()
            
            # Use episode index in the entity path to separate them
            # e.g. "episode_0/observation/images/top", "episode_1/observation/images/top"
            entity_prefix = f"episode_{ep_idx}"
            
            # We also need a timeline. Rerun defaults to a single global timeline.
            # If we want to scrub through all episodes independently, we might want to just map
            # frame_index on the timeline, but then they overlap.
            # Best approach for "watching one after another" is usually just logging sequentially
            # or mapping "index" to the global timeline.
            # However, batch["index"] is unique across the whole dataset so it works perfectly for sequential viewing.
            
            rr.set_time("global_index", sequence=batch["index"][i].item())
            
            # Also log frame_index per episode if we want to align them?
            # Let's stick to global index for "all at once" linear view, 
            # OR we can let user filter by entity in the viewer.
            
            rr.set_time("episode_time", sequence=batch["frame_index"][i].item())

            # display each camera image
            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                # Log under specific episode prefix
                rr.log(f"{entity_prefix}/{key}", entity=img_entity)

            # display action
            if ACTION in batch:
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{entity_prefix}/{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # display state
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"{entity_prefix}/state/{dim_idx}", rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(f"{entity_prefix}/{DONE}", rr.Scalars(batch[DONE][i].item()))

    if mode == "distant":
         try:
            while True:
                time.sleep(1)
         except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--episodes", type=str, required=True, help="Range of episodes, e.g. '0-10' or '0,1,3'")
    parser.add_argument("--display-compressed-images", type=int, default=1)
    
    args = parser.parse_args()
    
    # Parse episodes
    ep_list = []
    for part in args.episodes.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ep_list.extend(range(start, end + 1))
        else:
            ep_list.append(int(part))
            
    print(f"Visualizing episodes: {ep_list}")

    dataset = LeRobotDataset(
        args.repo_id, 
        root=args.root, 
        episodes=ep_list,
        tolerance_s=0.1 # loosen tolerance a bit if needed
    )

    visualize_dataset(
        dataset, 
        repo_id=args.repo_id,
        display_compressed_images=bool(args.display_compressed_images)
    )

if __name__ == "__main__":
    main()
