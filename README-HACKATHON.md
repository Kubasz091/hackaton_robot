# Policy Deployment Guide for YAM Arms

This guide explains how to deploy your trained policy on the bimanual YAM robot arms during the hackathon.

## Architecture Overview

Your policy runs inside a Docker container on our GPU server. The robot client (which we control) sends observations from the real robot to your policy server and executes the returned actions.


<pre>
┌──────────────────────────────────────────────────────────────────┐
│                           GPU Server                             │
│                                                                  │
│  ┌───────────────────────────┐                                   │
│  │     Docker Container      │                                   │
│  │  ┌─────────────────────┐  │                                   │
│  │  │    Policy Server    │  |◄─── You provide this              │
│  │  │    (your policy)    │  │                                   │
│  │  └─────────────────────┘  │                                   │
│  └─────────────┬─────────────┘                                   │
│                │ gRPC                                            │
│                ▼                                                 │
│  ┌─────────────────────────┐       ┌─────────────────────────┐   │
│  │      Robot Client       │◄─────►│        YAM Arms         │   │
│  │      (we run this)      │       │    (cameras + motors)   │   │
│  └─────────────────────────┘       └─────────────────────────┘   │
│                │                                                 │
│                │ HTTP                                            │
│                ▼                                                 │
│  ┌─────────────────────────┐                                     │
│  │       Visualizer        │◄── Also runnable on your machine    │
│  └─────────────────────────┘                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
</pre>


**Key points:**
- The robot client reads camera images and joint positions from the real YAM arms
- Your policy server receives observations and returns action chunks
- Communication happens over gRPC on port 7777
- The visualizer runs on the GPU server by default. You can also run it on your own machine during local development
- For safety, we first run with `--viz_only=true` to verify behavior before real deployment

## Deployment Workflow

1. **You prepare** a Docker image containing your policy inference code
2. **We run** the Docker container on our GPU server
3. **We launch** the policy server inside the container
4. **We run** the robot client that connects to your policy server
5. **Safety check** — we first run with `--viz_only=true` (actions shown in visualizer only)
6. **Real deployment** — if behavior looks correct, we deploy on the real YAM arms

## Preparing Your Docker Image

### Base Image

You can start from the official LeRobot GPU image:

```dockerfile
FROM huggingface/lerobot-gpu
```

### PyTorch Compatibility

Our server uses an NVIDIA B200 GPU. Install a compatible PyTorch version:

```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Model Weights

You have two options:
1. **Bake weights into the image** — include them in your Docker build
2. **Mount as volume** — copied from the cluster or your machine

### Example Docker Run Command

This is how we will run your container:

```bash
docker run -it --rm --gpus all -p 7777:7777 \
    -v /path/to/weights:/weights:ro \
    your-image-name /bin/bash
```

### Starting the Policy Server

Inside the container, start the policy server:

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=7777
```

## Saving and Loading Docker Images

To share your Docker image with us:

**Save to tar file:**
```bash
docker save -o my_policy_image.tar my-image-name
```

**Load from tar file:**
```bash
docker load -i my_policy_image.tar
```

## Local Development and Testing

You can utilize our visualizer to test your policy locally using pre-recorded observation data before deployment.

### Installation

We recommend using `uv` with Python 3.10:

```bash
git clone https://github.com/lute-corp/lerobot_hackathon
cd lerobot_hackathon

# Create environment
uv venv --python 3.10
source .venv/bin/activate

# Install with your policy's requirements (e.g., ACT)
uv pip install -e '.[act]'

# Install visualizer dependencies
uv pip install -r experimental/yam_visualization/requirements.txt
```

### Testing with Pre-Recorded Observations

You can run inference on pre-recorded initial state observation data without access to the real robot.

#### Step 1: Prepare Observation Data

Create a directory with the following structure:

```
observation_data/
├── top.png
├── wrist_left.png
├── wrist_right.png
└── motor_positions.json
```

The `motor_positions.json` file contains all 14 joint positions:

```json
{
    "joint1_left": 0.0,
    "joint2_left": 0.0,
    "joint3_left": 0.0,
    "joint4_left": 0.0,
    "joint5_left": 0.0,
    "joint6_left": 0.0,
    "gripper_left": 0.0,
    "joint1_right": 0.0,
    "joint2_right": 0.0,
    "joint3_right": 0.0,
    "joint4_right": 0.0,
    "joint5_right": 0.0,
    "joint6_right": 0.0,
    "gripper_right": 0.0
}
```

#### Step 2: Start the Visualizer

In terminal 1:

```bash
python experimental/yam_visualization/viz.py
```

Open `http://localhost:8081` in your browser to see the robot visualization.

#### Step 3: Start the Policy Server

In terminal 2:

```bash
python -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 \
    --port=7777
```

#### Step 4: Run Offline Inference

In terminal 3:

```bash
python experimental/send_offline_observation.py \
    --data_dir=/path/to/observation_data \
    --server_address=localhost:7777 \
    --policy_type=act \
    --pretrained_name_or_path=/path/to/weights \
    --task="plug the wire rope into the rubber cable" \
    --policy_device=cuda \
    --actions_per_chunk=100 \
    --viz_joints_address=127.0.0.1:5001
```

The predicted actions will loop in the visualizer. Press `Ctrl+C` to stop.

## Technical Reference

### Robot Client Command (Reference Only)

This is the command we run on our server. You do not need to run this yourself.
Argument values will be adjusted to your setup.

```bash
python -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:7777 \
    --robot.type=yam_follower_bimanual \
    --robot.cameras='{
        "top": {
            "type": "intelrealsense",
            "serial_number_or_name": "409122272894",
            "width": 640, "height": 480, "fps": 30
        },
        "wrist_left": {
            "type": "intelrealsense",
            "serial_number_or_name": "352122272837",
            "width": 640, "height": 480, "fps": 30
        },
        "wrist_right": {
            "type": "intelrealsense",
            "serial_number_or_name": "352122272507",
            "width": 640, "height": 480, "fps": 30
        }
    }' \
    --task="plug the wire rope into the rubber cable" \
    --policy_type=act \
    --actions_per_chunk=100 \
    --pretrained_name_or_path=/weights \
    --chunk_size_threshold=1.0 \
    --aggregate_fn_name=weighted_average \
    --policy_device=cuda \
    --viz_joints_address=127.0.0.1:5001 \
    --viz_only=true
```

**Key parameters:**
- `--viz_only=true` — actions are sent to visualizer only, not to real motors
- `--viz_joints_address` — address where the visualizer receives joint positions
- `--actions_per_chunk` — number of actions returned per inference call
- `--chunk_size_threshold` — controls when new observations are sent (1.0 = send immediately after actions run out)
