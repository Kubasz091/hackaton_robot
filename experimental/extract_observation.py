
import argparse
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def main():
    parser = argparse.ArgumentParser(description="Extract observation data manually.")
    parser.add_argument("--repo-id", type=str, default="yam_wire", help="Ignored if root is set")
    parser.add_argument("--root", type=Path, default="yam_wire", help="Path to local dataset root")
    parser.add_argument("--output-dir", type=Path, default="observation_data", help="Output directory.")
    
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching in root: {root}")
    
    # 1. Image Extraction
    videos_dir = root / "videos"
    if not videos_dir.exists():
        print(f"Error: {videos_dir} does not exist.")
        return

    expected_cameras = ["top", "wrist_left", "wrist_right"]
    
    for cam_name in expected_cameras:
        dataset_key = f"observation.images.{cam_name}"
        cam_dir = videos_dir / dataset_key
        
        extracted = False
        if cam_dir.exists():
            # Find first mp4
            mp4_files = sorted(list(cam_dir.glob("**/*.mp4")))
            if mp4_files:
                video_path = mp4_files[0]
                print(f"Reading {video_path}...")
                
                # Try using ffmpeg command line directly
                import subprocess
                out_path = output_dir / f"{cam_name}.png"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(out_path)
                ]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"Saved {out_path} (via ffmpeg)")
                    extracted = True
                except subprocess.CalledProcessError as e:
                    print(f"ffmpeg failed for {video_path}: {e}")
                    # Fallback to cv2 if needed, but likely won't work if ffmpeg CLI failed or if cv2 failed before
            else:
                print(f"No mp4 files found in {cam_dir}")
        else:
            print(f"Directory {cam_dir} not found.")
            
        if not extracted:
            # Generate black image
            print(f"Generating blank image for {cam_name}")
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            out_path = output_dir / f"{cam_name}.png"
            cv2.imwrite(str(out_path), blank)

    # 2. Motor Positions Extraction
    data_dir = root / "data"
    chunk_0 = data_dir / "chunk-000"
    parquet_files = sorted(list(chunk_0.glob("*.parquet")))
    
    if parquet_files:
        p_path = parquet_files[0]
        print(f"Reading data from {p_path}...")
        df = pd.read_parquet(p_path)
        
        if "observation.state" in df.columns:
            # Get first row
            state = df.iloc[0]["observation.state"]
            # state is numpy array typically
            
            if len(state) == 14:
                print("Found 14 joints. Mapping to standard names...")
                # Mapping assumption:
                # 0-6: left arm (j1..j6, gripper)
                # 7-13: right arm (j1..j6, gripper)
                
                mapped = {
                    "joint1_left": float(state[0]),
                    "joint2_left": float(state[1]),
                    "joint3_left": float(state[2]),
                    "joint4_left": float(state[3]),
                    "joint5_left": float(state[4]),
                    "joint6_left": float(state[5]),
                    "gripper_left": float(state[6]),
                    
                    "joint1_right": float(state[7]),
                    "joint2_right": float(state[8]),
                    "joint3_right": float(state[9]),
                    "joint4_right": float(state[10]),
                    "joint5_right": float(state[11]),
                    "joint6_right": float(state[12]),
                    "gripper_right": float(state[13]),
                }
                
                json_path = output_dir / "motor_positions.json"
                with open(json_path, 'w') as f:
                    json.dump(mapped, f, indent=4)
                print(f"Saved {json_path}")
            else:
                print(f"Warning: Expected 14 joints, found {len(state)}. Dumping generic.")
                # fallback
                mapped = {f"joint_{i}": float(v) for i, v in enumerate(state)}
                with open(output_dir / "motor_positions.json", 'w') as f:
                    json.dump(mapped, f, indent=4)
        else:
            print("Error: 'observation.state' column not found in parquet.")
            print("Columns:", df.columns)
    else:
        print(f"No parquet files found in {chunk_0}")

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
