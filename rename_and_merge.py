from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.dataset_tools import merge_datasets # unused but ok
from lerobot.datasets.utils import (
    load_episodes, 
    DEFAULT_DATA_PATH, 
    write_info, 
    write_stats, 
    write_tasks
)
from lerobot.utils.utils import init_logging
import logging
import shutil
import pandas as pd
from tqdm import tqdm

def rename_dataset_features(src_root: Path, dst_root: Path, rename_map: dict, repo_id: str):
    """
    Creates a copy of the dataset at src_root to dst_root with features renamed according to rename_map.
    """
    logging.info(f"Renaming features from {src_root} to {dst_root} with map: {rename_map}")
    
    # Load source dataset
    # Using dummy ID, force valid root
    ds = LeRobotDataset("dummy_source", root=src_root)
    
    # 1. Prepare new features dict
    new_features = {}
    for k, v in ds.meta.features.items():
        if k in rename_map:
            new_features[rename_map[k]] = v
        else:
            new_features[k] = v
            
    # 2. Create Destination Metadata
    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=ds.meta.fps,
        features=new_features,
        robot_type=ds.meta.robot_type,
        root=dst_root,
        use_videos=len(ds.meta.video_keys) > 0,
        chunks_size=ds.meta.chunks_size,
    )
    # Ensure root exists
    dst_meta.root.mkdir(parents=True, exist_ok=True)
    
    # 3. Copy/Rename Data (Parquet)
    if ds.meta.episodes is None:
        ds.meta.episodes = load_episodes(ds.meta.root)
        
    file_to_episodes = {}
    for ep in ds.meta.episodes:
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        key = (chunk_idx, file_idx)
        if key not in file_to_episodes:
            file_to_episodes[key] = []
        file_to_episodes[key].append(ep)

    for (chunk_idx, file_idx), _ in tqdm(file_to_episodes.items(), desc="Renaming parquet columns"):
        src_parquet = ds.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        dst_parquet = dst_meta.root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.read_parquet(src_parquet)
        df.rename(columns=rename_map, inplace=True)
        df.to_parquet(dst_parquet, index=False)

    # 4. Copy/Rename Videos
    for old_key in ds.meta.video_keys:
        new_key = rename_map.get(old_key, old_key)
        src_vid_dir = ds.root / "videos" / old_key
        dst_vid_dir = dst_meta.root / "videos" / new_key
        
        if src_vid_dir.exists():
            if dst_vid_dir.exists():
                shutil.rmtree(dst_vid_dir)
            shutil.copytree(src_vid_dir, dst_vid_dir)
            
    # 5. Metadata Update (Episodes)
    ep_files = sorted((ds.root / "meta/episodes").glob("*/*.parquet"))
    
    if not ep_files:
        logging.warning("No episode metadata files found in source meta/episodes!")
        
    for src_ep_path in ep_files:
        rel_path = src_ep_path.relative_to(ds.root)
        dst_ep_path = dst_meta.root / rel_path
        dst_ep_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_ep = pd.read_parquet(src_ep_path)
        new_cols = {}
        for col in df_ep.columns:
            for old_name, new_name in rename_map.items():
                if old_name in col:
                     new_cols[col] = col.replace(old_name, new_name)
        df_ep.rename(columns=new_cols, inplace=True)
        df_ep.to_parquet(dst_ep_path, index=False)

    # 5b. Update Info totals explicitly
    dst_meta.info["total_episodes"] = ds.meta.total_episodes
    dst_meta.info["total_frames"] = ds.meta.total_frames
    dst_meta.info["total_tasks"] = ds.meta.total_tasks
    dst_meta.info["splits"] = ds.meta.info.get("splits", {})
    
    # 6. Copy Stats and Rename
    if ds.meta.stats:
        new_stats = {}
        for k, v in ds.meta.stats.items():
            if k in rename_map:
                new_stats[rename_map[k]] = v
            else:
                new_stats[k] = v
        write_stats(new_stats, dst_meta.root)
        
    # 7. Copy Tasks
    if ds.meta.tasks is not None:
        write_tasks(ds.meta.tasks, dst_meta.root)
        
    # 8. Write Info
    write_info(dst_meta.info, dst_meta.root)
    
    logging.info(f"Successfully renamed dataset to {dst_root}")
    # We do NOT return a loaded dataset object here because it triggers Hub checks
    return dst_root


def main():
    init_logging()
    
    # --- Step 1: Rename ---
    rename_map = {
        "observation.images.top": "observation.images.wrist_right",
        "observation.images.wrist_left": "observation.images.top",
        "observation.images.wrist_right": "observation.images.wrist_left"
    }
    
    src_root = Path("/common/yam_wire")
    tmp_root = Path("/home/hacker/lerobot_hackathon/yam_wire_tmp")
    
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
        
    # Create temp dataset on disk
    rename_dataset_features(
        src_root=src_root,
        dst_root=tmp_root,
        rename_map=rename_map,
        repo_id="lute/yam_wire_tmp"
    )
    
    # --- Step 2: Merge using aggregate_datasets ---
    
    output_repo_id = "lute/yam_wire_merged"
    output_root = Path("/home/hacker/lerobot_hackathon/yam_wire_merged")
    
    if output_root.exists():
        shutil.rmtree(output_root)
        
    logging.info("Merging yam_wire_tmp and yam_wire_2...")
    
    # Use direct aggregation passing paths
    repo_ids = ["lute/yam_wire_tmp", "lute/yam_wire_2"]
    roots = [tmp_root, Path("/common/yam_wire_2")]
    
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=output_repo_id,
        roots=roots,
        aggr_root=output_root
    )
    
    logging.info(f"Merge complete! Output at {output_root}")
    
    # Verify result safely
    # We can inspect info.json manually
    import json
    with open(output_root / "meta/info.json") as f:
        info = json.load(f)
        logging.info(f"Available merged episodes: {info['total_episodes']}")

if __name__ == "__main__":
    main()
