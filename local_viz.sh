#!/bin/bash
# Script to run local visualization
# Validated against README-HACKATHON.md

# Activate virtual environment
source .venv/bin/activate

# Function to kill background jobs on exit
cleanup() {
    echo "Stopping background processes..."
    kill $(jobs -p)
}
trap cleanup EXIT

echo "Starting Visualizer (http://localhost:8081)..."
python experimental/yam_visualization/viz.py &

# Default to enabling ensembling
ENSEMBLING_FLAG="--use_trajectory_ensembling=true"

# Parse arguments
for arg in "$@"
do
    case $arg in
        --no-ensemble)
        ENSEMBLING_FLAG="--use_trajectory_ensembling=false"
        shift
        ;;
    esac
done

echo "Starting Policy Server (Ensembling: $ENSEMBLING_FLAG)..."
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=7777 \
    $ENSEMBLING_FLAG &

echo "Waiting 5 seconds for services to start..."
sleep 5

echo "Starting Offline Inference..."
./test.sh

wait
