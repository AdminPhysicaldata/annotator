"""Create a synthetic test dataset for VIVE Labeler.

This script generates a simple synthetic dataset with:
- Synthetic video (moving colored circle)
- VIVE tracking data (circular trajectory)
- LeRobot v3.0 compatible format

Usage:
    python scripts/create_test_dataset.py --output-dir ./test_dataset
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm


def generate_circular_trajectory(num_frames, radius=0.5, height=1.0, fps=30.0):
    """Generate circular trajectory for VIVE tracker.

    Args:
        num_frames: Number of frames
        radius: Radius of circle (meters)
        height: Height above ground (meters)
        fps: Frames per second

    Returns:
        timestamps, positions, rotations
    """
    timestamps = np.arange(num_frames) / fps

    # Circular trajectory in XY plane
    angles = np.linspace(0, 4 * np.pi, num_frames)  # 2 full rotations

    positions = np.zeros((num_frames, 3))
    positions[:, 0] = radius * np.cos(angles)  # X
    positions[:, 1] = radius * np.sin(angles)  # Y
    positions[:, 2] = height + 0.1 * np.sin(2 * angles)  # Z with slight wave

    # Rotations (quaternions) - rotating around Z axis
    rotations = np.zeros((num_frames, 4))
    rotations[:, 0] = np.cos(angles / 2)  # w
    rotations[:, 1] = 0.0  # x
    rotations[:, 2] = 0.0  # y
    rotations[:, 3] = np.sin(angles / 2)  # z

    return timestamps, positions, rotations


def create_synthetic_video(output_path, num_frames, width=640, height=480, fps=30.0):
    """Create synthetic video with moving colored circle.

    Args:
        output_path: Path to output MP4 file
        num_frames: Number of frames
        width: Video width
        height: Video height
        fps: Frames per second
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i in tqdm(range(num_frames), desc="Generating video"):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray background

        # Moving circle
        angle = 2 * np.pi * i / num_frames
        cx = int(width / 2 + width / 3 * np.cos(angle))
        cy = int(height / 2 + height / 3 * np.sin(angle))

        # Draw circle with changing color
        color = (
            int(127 + 127 * np.cos(angle)),
            int(127 + 127 * np.sin(angle)),
            int(127 + 127 * np.cos(angle + np.pi/2))
        )
        cv2.circle(frame, (cx, cy), 30, color, -1)

        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add trajectory trace
        if i > 0:
            for j in range(max(0, i - 50), i):
                prev_angle = 2 * np.pi * j / num_frames
                px = int(width / 2 + width / 3 * np.cos(prev_angle))
                py = int(height / 2 + height / 3 * np.sin(prev_angle))
                alpha = (j - max(0, i - 50)) / 50
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

        out.write(frame)

    out.release()
    print(f"Video created: {output_path}")


def create_test_dataset(output_dir, num_episodes=2, frames_per_episode=300, fps=30.0):
    """Create complete test dataset.

    Args:
        output_dir: Output directory path
        num_episodes: Number of episodes to generate
        frames_per_episode: Frames per episode
        fps: Frames per second
    """
    output_dir = Path(output_dir)

    print(f"Creating test dataset in: {output_dir}")

    # Create directory structure
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "front_camera").mkdir(parents=True, exist_ok=True)

    # Generate episodes
    all_data = []
    episodes_metadata = []
    frame_offset = 0

    for ep_idx in range(num_episodes):
        print(f"\n=== Generating Episode {ep_idx} ===")

        episode_id = f"episode_{ep_idx:06d}"

        # Generate trajectory
        timestamps, positions, rotations = generate_circular_trajectory(
            num_frames=frames_per_episode,
            radius=0.5 + 0.2 * ep_idx,  # Vary radius per episode
            height=1.0,
            fps=fps
        )

        # Generate video
        video_path = output_dir / "videos" / "front_camera" / f"{episode_id}.mp4"
        create_synthetic_video(
            output_path=video_path,
            num_frames=frames_per_episode,
            fps=fps
        )

        # Create dataframe for this episode
        episode_data = {
            'episode_index': [ep_idx] * frames_per_episode,
            'frame_index': np.arange(frame_offset, frame_offset + frames_per_episode),
            'timestamp': timestamps,
            'observation.vive_position.x': positions[:, 0],
            'observation.vive_position.y': positions[:, 1],
            'observation.vive_position.z': positions[:, 2],
            'observation.vive_rotation.w': rotations[:, 0],
            'observation.vive_rotation.x': rotations[:, 1],
            'observation.vive_rotation.y': rotations[:, 2],
            'observation.vive_rotation.z': rotations[:, 3],
        }

        df_episode = pd.DataFrame(episode_data)
        all_data.append(df_episode)

        # Episode metadata
        episodes_metadata.append({
            'episode_id': episode_id,
            'episode_index': ep_idx,
            'from': frame_offset,
            'to': frame_offset + frames_per_episode,
            'duration': frames_per_episode / fps,
            'fps': fps,
        })

        frame_offset += frames_per_episode

    # Combine all episodes
    print("\n=== Saving dataset files ===")

    df_all = pd.concat(all_data, ignore_index=True)
    parquet_path = output_dir / "data" / "chunk_000000.parquet"
    df_all.to_parquet(parquet_path, index=False)
    print(f"Saved sensor data: {parquet_path}")

    # Save episode metadata
    df_episodes = pd.DataFrame(episodes_metadata)
    episodes_path = output_dir / "meta" / "episodes" / "episodes.parquet"
    df_episodes.to_parquet(episodes_path, index=False)
    print(f"Saved episodes metadata: {episodes_path}")

    # Create info.json
    info = {
        "dataset_type": "LeRobotDataset",
        "version": "3.0",
        "fps": fps,
        "camera_keys": ["front_camera"],
        "features": {
            "observation.vive_position": {"dtype": "float32", "shape": [3]},
            "observation.vive_rotation": {"dtype": "float32", "shape": [4]},
            "timestamp": {"dtype": "float64", "shape": []},
        },
        "total_episodes": num_episodes,
        "total_frames": frame_offset,
        "description": "Synthetic test dataset with circular VIVE trajectory",
    }

    info_path = output_dir / "meta" / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved dataset info: {info_path}")

    # Create stats.json
    stats = {
        "observation.vive_position": {
            "mean": df_all[['observation.vive_position.x',
                          'observation.vive_position.y',
                          'observation.vive_position.z']].mean().tolist(),
            "std": df_all[['observation.vive_position.x',
                         'observation.vive_position.y',
                         'observation.vive_position.z']].std().tolist(),
            "min": df_all[['observation.vive_position.x',
                         'observation.vive_position.y',
                         'observation.vive_position.z']].min().tolist(),
            "max": df_all[['observation.vive_position.x',
                         'observation.vive_position.y',
                         'observation.vive_position.z']].max().tolist(),
        }
    }

    stats_path = output_dir / "meta" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_path}")

    # Summary
    print("\n" + "=" * 60)
    print("✅ Test dataset created successfully!")
    print("=" * 60)
    print(f"Location: {output_dir.absolute()}")
    print(f"Episodes: {num_episodes}")
    print(f"Total frames: {frame_offset}")
    print(f"Duration per episode: {frames_per_episode / fps:.1f}s")
    print(f"Total duration: {frame_offset / fps:.1f}s")
    print("\nTo use in VIVE Labeler:")
    print(f"  1. Launch: python -m vive_labeler")
    print(f"  2. File → Open Dataset")
    print(f"  3. Select: {output_dir.absolute()}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic test dataset for VIVE Labeler"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_dataset",
        help="Output directory for dataset (default: ./test_dataset)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes (default: 2)"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="Frames per episode (default: 300)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second (default: 30.0)"
    )

    args = parser.parse_args()

    create_test_dataset(
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        frames_per_episode=args.frames,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
