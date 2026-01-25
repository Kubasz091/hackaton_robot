#!/bin/bash
# Usage: ./run_viz.sh [dataset_name] [episodes_range]
# Example: ./run_viz.sh yam_wire 0-10

DATASET=${1:-yam_wire}
EPISODES=${2:-0-10}

echo "Visualizing dataset: $DATASET, episodes: $EPISODES"
python viz_episodes.py --repo-id "$DATASET" --root "$DATASET" --episodes "$EPISODES"
