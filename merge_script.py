from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.utils.utils import init_logging
import logging

def main():
    init_logging()
    
    # Define source datasets with their specific roots
    # Note: repo_id is likely just the last part of the path or the folder name in a local setup
    # but based on user request "lute/yam_wire", we use that.
    # The root is where the "lute" folder is located.
    ds1 = LeRobotDataset("lute/yam_wire", root="/common/yam_wire")
    ds2 = LeRobotDataset("lute/yam_wire_2", root="/common/yam_wire_2")
    
    # Output configuration
    output_repo_id = "lute/yam_wire_merged"
    output_root = Path("/home/hacker/lerobot_hackathon/yam_wire_merged")
    
    # The merge function logic expects output_dir to be the full path including repo_id structure
    # if it follows standard HF cache structure, but here we likely want it flat or specific.
    # Let's look at how handle_merge did it: 
    # output_dir = Path(cfg.root) / cfg.repo_id
    output_dir = output_root  # We want the dataset AT this path directly
    
    logging.info(f"Merging datasets...")
    merge_datasets(
        [ds1, ds2],
        output_repo_id=output_repo_id,
        output_dir=output_dir,
    )
    logging.info(f"Successfully merged to {output_dir}")

if __name__ == "__main__":
    main()
