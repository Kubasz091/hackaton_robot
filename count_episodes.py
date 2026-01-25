
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import sys

def count_episodes(repo_id):
    # Adjust root path logic to match local structure
    root = Path.cwd() / repo_id
    
    print(f"Loading dataset from: {root}")
    if not root.exists():
        print(f"Error: Directory {root} does not exist.")
        return

    try:
        # Load dataset
        dataset = LeRobotDataset(root=root, repo_id=repo_id)
        print(f"Dataset: {repo_id}")
        print(f"Total episodes: {dataset.meta.total_episodes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_episodes.py <dataset_name>")
        # Default to yam_wire for convenience if no arg provided
        print("Example: python count_episodes.py yam_wire")
    else:
        count_episodes(sys.argv[1])
