
import cv2
import numpy as np
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def compare_datasets(dataset1_path: str, dataset2_path: str, output_path: str = "camera_comparison.jpg"):
    root_path = Path.cwd()
    print(f"Using root: {root_path}")
    
    # Check if directories exist
    if not (root_path / dataset1_path).exists():
        print(f"Error: {root_path / dataset1_path} does not exist")
        return
        
    print(f"Loading dataset 1: {dataset1_path} from {root_path / dataset1_path}")
    ds1 = LeRobotDataset(root=root_path / dataset1_path, repo_id=dataset1_path, episodes=[0])
    
    print(f"Loading dataset 2: {dataset2_path} from {root_path / dataset2_path}")
    ds2 = LeRobotDataset(root=root_path / dataset2_path, repo_id=dataset2_path, episodes=[0])
    
    # Get frame 0 from both
    item1 = ds1[0]
    item2 = ds2[0]
    
    # Identify camera keys
    camera_keys = [k for k in item1.keys() if "image" in k]
    print(f"Camera keys: {camera_keys}")
    
    rows = []
    
    for key in camera_keys:
        # Get images (C, H, W) float32 -> (H, W, C) uint8
        img1 = (item1[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img2 = (item2[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Add label
        cv2.putText(img1, f"{Path(dataset1_path).name} - {key}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img2, f"{Path(dataset2_path).name} - {key}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Concatenate horizontally
        row = np.hstack([img1, img2])
        rows.append(row)
        
    # Concatenate all rows vertically
    final_image = np.vstack(rows)
    
    cv2.imwrite(output_path, final_image)
    print(f"Comparison image saved to {output_path}")

if __name__ == "__main__":
    compare_datasets("yam_wire", "yam_wire_2")
